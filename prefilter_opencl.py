import argparse
import pyopencl as cl
import pyopencl.array as cl_array
import shutil
import sys
import os
import math
import numpy
import time
import OpenImageIO as oiio


def print_device(device):
    print device.name
    print device.extensions
    print "Memory {} MB".format(device.global_mem_size / (1024 * 1024))
    print "Work group {}".format(device.max_work_group_size)


class Prefilter:

    def __init__(self, input, **kwargs):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        # platform = cl.get_platforms()[0]
        # device_list = platform.get_devices(cl.device_type.GPU)
        device = self.ctx.devices[0]
        print_device(device)

        self.input_filename = input
        # self.input_filename = "/tmp/specular_%d.tif"

        self.original_cubemap_size = None
        self.d_cubemap_levels = None
        self.samples = []
        self.init_data()

    def load_program(self, filename):
        # read in the OpenCL source file as a string
        python_dirname = os.path.dirname(os.path.realpath(__file__))
        f = open(os.path.join(python_dirname, filename), 'r')
        fstr = "".join(f.readlines())

        # inject dymanic argument
        num_lod = len(self.d_cubemap_levels)
        mipmap_arguments = "read_only image2d_array_t cubemap0\n"
        get_sample_implementation = "static float4 getSample(float level, float4 uv, read_only image2d_array_t cubemap0 "
        for i in range(1, num_lod):
            arg = ",read_only image2d_array_t cubemap{}".format(i)
            mipmap_arguments += arg
            get_sample_implementation += arg
        get_sample_implementation += ") {\n"

        if num_lod > 1:
            get_sample_implementation += "float r;\nfloat4 color0,color1;\n"
            for i in range(num_lod - 1):
                lstr = "if (level < {}.0f)".format(i + 1)
                lstr += " {\n"
                lstr += "  r = level - {}.0f;\n".format(i)
                lstr += "  color0 = read_imagef( cubemap{}, cubemapSampler, uv );\n".format(i)
                lstr += "  color1 = read_imagef( cubemap{}, cubemapSampler, uv );\n".format(i + 1)
                lstr += "  return mix(color0, color1, r);\n}\n"
                get_sample_implementation += lstr
            get_sample_implementation += "return read_imagef( cubemap{}, cubemapSampler, uv );\n".format(num_lod - 1)
        else:
            get_sample_implementation += "return read_imagef( cubemap0, cubemapSampler, uv );\n"
        get_sample_implementation += "}\n"

        fstr = fstr.replace("GET_SAMPLE_IMPLEMENTATION", get_sample_implementation)

        # print mipmap_arguments
        fstr = fstr.replace("MIPMAP_LEVEL_ARGUMENTS", mipmap_arguments)

        # inject dymanic declaration
        get_sample_call = "getSample(lod, uv, cubemap0"
        for i in range(1, num_lod):
            get_sample_call += ", cubemap{}".format(i)
        get_sample_call += ");"

        # print get_sample_call
        fstr = fstr.replace("GET_SAMPLE_CALL", get_sample_call)
        #print fstr
        # create the program
        self.program = cl.Program(self.ctx, fstr).build()

    def init_data(self):

        cubemap_filenames = []
        if '%' in self.input_filename:
            for level in range(30):
                filename = self.input_filename % level
                if not os.path.isfile(filename):
                    break
                cubemap_filenames.append(filename)
        else:
            cubemap_filenames.append(self.input_filename)

        cubemap_levels = []
        cubemap_levels_info = []

        for filename in cubemap_filenames:
            cubemap = read_image_cubemap(filename)
            size = cubemap[0].shape[0]
            d_cubemap = self.d_create_image_cubemap(cubemap)
            cubemap_levels.append(d_cubemap)
            cubemap_levels_info.append({"size": size,
                                        "filename": filename,
                                        "d_cubemap": d_cubemap})

        self.d_cubemap_levels = cubemap_levels
        self.cubemap_levels = cubemap_levels_info
        self.original_cubemap_size = cubemap_levels_info[0]["size"]

        self.load_program("ggx.cl")

    def init_specular_filtering(self, size, num_samples, sample_file, nb_levels):

        self.h_precomputed_light = numpy.zeros(num_samples, dtype=cl_array.vec.float4)
        self.h_nol = numpy.zeros((num_samples), dtype=numpy.float32)
        self.h_current_light_index = numpy.zeros((1), dtype=numpy.uint32)

        if sample_file:
            print "reading samples from file {} with {} levels ".format(sample_file, nb_levels)
            self.samples = []
            f = open(sample_file, 'rb')
            for i in range(nb_levels):
                array = numpy.fromfile(f, dtype=cl_array.vec.float4, count=num_samples)
                self.samples.append(array)
        else:
            self.d_precomputed_lightvector_write = cl.Buffer(self.ctx,
                                                             cl.mem_flags.WRITE_ONLY,
                                                             num_samples * 4 * 4)

        self.d_precomputed_lightvector_read = cl.Buffer(self.ctx,
                                                        cl.mem_flags.READ_ONLY,
                                                        num_samples * 4 * 4)

        self.d_current_light_index = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.h_current_light_index)

        self.d_nol = cl.Buffer(self.ctx,
                               cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_nol)

    def get_sequence(self, level, roughness_linear, num_samples):

        # predefined samples from file
        if len(self.samples):
            self.h_precomputed_light = self.samples[level-1]
            sum_weight = 0.0
            for i in self.h_precomputed_light:
                sum_weight += i[2]

            return {
                'sum': sum_weight,
                'tries': 1,
                'sequence': num_samples
            }

        try_sequence = num_samples
        h_precomputed_lightvector = self.h_precomputed_light

        h_nol = self.h_nol
        d_nol = self.d_nol

        d_current_light_index = self.d_current_light_index
        h_current_light_index = self.h_current_light_index

        nb_try = 0
        while True:
            nb_try += 1

            # reset the buffers
            h_current_light_index.fill(0)
            cl.enqueue_copy(self.queue,
                            d_current_light_index,
                            h_current_light_index)

            h_nol.fill(0.0)
            cl.enqueue_copy(self.queue,
                            d_nol,
                            h_nol).wait()

            # create an image to get the result
            # it should be full mipmap chain
            event = self.program.computeLightVector(self.queue,
                                                    (try_sequence,),
                                                    None,
                                                    self.d_precomputed_lightvector_write,
                                                    d_nol,
                                                    d_current_light_index,
                                                    numpy.uint32(try_sequence),
                                                    numpy.uint32(num_samples),
                                                    numpy.uint32(self.original_cubemap_size),
                                                    numpy.float32(roughness_linear))

            cl.enqueue_copy(self.queue, h_current_light_index, d_current_light_index).wait()
            nb_valid_samples = h_current_light_index[0]
            # print "computeLightVector duration {} valids {} / {}".format(
            #     1e-9*(event.profile.end - event.profile.start), nb_valid_samples, try_sequence)
            if nb_valid_samples != num_samples:
                try_sequence += num_samples - nb_valid_samples
                continue

            break

        cl.enqueue_copy(self.queue,
                        h_precomputed_lightvector,
                        self.d_precomputed_lightvector_write).wait()
        # for i in h_precomputed_lightvector:
        #     print i[0], i[1], i[2], i[3]

        cl.enqueue_copy(self.queue, h_nol, d_nol).wait()

        sum_weight = numpy.add.reduce(h_nol.astype(numpy.double))
        # print "compute sequence in {} tries with {} sequence".format(nb_try, try_sequence)

        return {
            'sum': sum_weight,
            'tries': nb_try,
            'sequence': try_sequence
        }

    def compute_level(self, level, size, roughness_linear, filename, num_samples, fix_edge):

        start_tick = time.time()

        num_light_vector = num_samples
        total_weight = 0.0

        if roughness_linear != 0.0:
            sequence = self.get_sequence(level, roughness_linear, num_samples)
            print "compute sequence in {} attempts, sequence {} for roughness {} : {} seconds".format(
                sequence['tries'],
                sequence['sequence'],
                roughness_linear,
                (time.time() - start_tick))
            total_weight = sequence['sum']
        else:
            self.h_precomputed_light = numpy.zeros(num_samples, dtype=cl_array.vec.float4)
            total_weight = 1.0
            num_light_vector = 1
            # force lod 0
            self.h_precomputed_light[0][0] = 0.0
            self.h_precomputed_light[0][1] = 0.0
            self.h_precomputed_light[0][2] = 1.0
            self.h_precomputed_light[0][3] = 0.0

        # copy for the buffer in readonly
        cl.enqueue_copy(self.queue,
                        self.d_precomputed_lightvector_read,
                        self.h_precomputed_light).wait()
        # print self.h_precomputed_light
        # print total_weight

        sys.stdout.write("compute ggx size {} roughness {} - ".format(size, roughness_linear))
        sys.stdout.flush()
        cubemap_result = []
        start_tick = time.time()
        event = None
        for face in range(6):
            # print "Face {}".format(face)
            h_face_result = numpy.zeros((size, size), dtype=cl_array.vec.float4)
            d_face_result = cl.Image(self.ctx,
                                     cl.mem_flags.WRITE_ONLY,
                                     cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                                     (size, size))

            # create an image to get the result
            # it should be full mipmap chain
            event = self.program.computeGGX(self.queue,
                                            (size, size),
                                            None,
                                            numpy.uint32(face),
                                            d_face_result,
                                            self.d_precomputed_lightvector_read,
                                            numpy.float32(total_weight),
                                            numpy.uint32(num_light_vector),
                                            numpy.uint32(1 if fix_edge else 0),
                                            *self.d_cubemap_levels)
            cl.enqueue_copy(self.queue,
                            h_face_result,
                            d_face_result,
                            origin=(0, 0, 0),
                            region=(size, size, 1)).wait()
            self.queue.finish()

            cubemap_result.append(h_face_result)
            sys.stdout.write("{:5.3f}s ".format(1e-9 * (event.profile.end - event.profile.start)))
            sys.stdout.flush()

        # face_size = size * size * 4 * 4
        # print "size {}".format(size)
        # for a in range(6):
        #     offset_start = a * face_size
        #     face_buffer = numpy.frombuffer(numpy.getbuffer(h_face_result, offset_start, face_size), dtype=cl_array.vec.float4)
        #     face_buffer = face_buffer.reshape(size, size)
        #     print face_buffer[0][0]
        #     cm.append(face_buffer)
        write_image_cubemap(cubemap_result, filename)

        sys.stdout.write(" : {} seconds\n".format((time.time() - start_tick)))
        sys.stdout.flush()

    def get_background_sequence(self, radius, num_samples):
        # 3*sigma rules
        sigma = radius / 3.0
        sigmaSqr = sigma * sigma

        d_precomputed_tap = cl.Buffer(self.ctx,
                                      cl.mem_flags.READ_WRITE,
                                      num_samples * 4 * 4)
        h_weight = numpy.zeros((num_samples), dtype=numpy.float32)
        d_weight = cl.Buffer(self.ctx,
                             cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=h_weight)

        event = self.program.computeTapVector(self.queue,
                                              (num_samples,),
                                              None,
                                              d_precomputed_tap,
                                              d_weight,
                                              numpy.float32(sigmaSqr),
                                              numpy.float32(radius),
                                              numpy.uint32(num_samples))

        cl.enqueue_copy(self.queue, h_weight, d_weight).wait()

        sum_weight = numpy.add.reduce(h_weight.astype(numpy.double))
        #print h_weight
        return {
            'sum': sum_weight,
            'tap_vector': d_precomputed_tap
        }

    def run_background_blur(self, output_file, **kwargs):
        size = kwargs.get("size")
        num_samples = kwargs.get("num_samples")
        radius = kwargs.get("radius")

        start_tick = time.time()

        cm_levels = self.cubemap_levels
        level = [f for f in cm_levels if f["size"] == size]
        d_cubemap_with_same_output_size = level[0]["d_cubemap"] if level else cm_levels[0]["d_cubemap"]

        filename = output_file
        sequence = self.get_background_sequence(radius, num_samples)
        print "compute background samples, sequence {} for radius {} : {} seconds".format(
            num_samples,
            radius,
            (time.time() - start_tick))

        d_tap_vector = sequence['tap_vector']
        total_weight = sequence['sum']

        sys.stdout.write("compute average blur size {} radius {} - ".format(size, radius))
        sys.stdout.flush()
        cubemap_result = []
        start_tick = time.time()
        event = None
        for face in range(6):
            # print "Face {}".format(face)
            h_face_result = numpy.zeros((size, size), dtype=cl_array.vec.float4)
            d_face_result = cl.Image(self.ctx,
                                     cl.mem_flags.WRITE_ONLY,
                                     cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                                     (size, size))

            # create an image to get the result
            # it should be full mipmap chain
            event = self.program.computeBackground(self.queue,
                                                   (size, size),
                                                   None,
                                                   numpy.uint32(face),
                                                   d_face_result,
                                                   d_tap_vector,
                                                   numpy.float32(total_weight),
                                                   numpy.uint32(num_samples),
                                                   numpy.uint32(1 if kwargs.get("fix_edge") else 0),
                                                   d_cubemap_with_same_output_size)
            cl.enqueue_copy(self.queue,
                            h_face_result,
                            d_face_result,
                            origin=(0, 0, 0),
                            region=(size, size, 1)).wait()
            self.queue.finish()

            cubemap_result.append(h_face_result)
            sys.stdout.write("{:5.3f}s ".format(1e-9 * (event.profile.end - event.profile.start)))
            sys.stdout.flush()

        write_image_cubemap(cubemap_result, filename)

        sys.stdout.write(" : {} seconds\n".format((time.time() - start_tick)))
        sys.stdout.flush()

    def run_ggx(self, output_directory, **kwargs):
        size = kwargs.get("size")
        num_samples = kwargs.get("num_samples")
        limit_size = kwargs.get("limit_size")

        sample_file = kwargs.get("sample_file", None)

        compute_start_size = size
        total_mipmap = math.log(compute_start_size) / math.log(2)
        end_mipmap = total_mipmap - math.log(limit_size) / math.log(2)

        nb_levels = int(end_mipmap)
        self.init_specular_filtering(size, num_samples, sample_file, nb_levels)

        cm_levels = self.cubemap_levels
        # keep a reference to the image that will be used as roughness 0
        level = [f for f in cm_levels if f['size'] == size]
        filename_image_roughness0 = level[0]["filename"] if level else None

        output = output_directory

        total_mipmap = int(total_mipmap)
        print "{} mipmaps levels will be generated from {}x{} to {}x{}".format(
            end_mipmap + 1, compute_start_size, compute_start_size, limit_size, limit_size)

        pattern_output = "{}_{}.tif"

        start_mipmap = 0
        # roughness 0 is a simple copy if found in mipmap files
        if filename_image_roughness0:
            start_mipmap = 1
            shutil.copyfile(filename_image_roughness0, pattern_output.format(output, 0))

        start = 0.0
        stop = 1.0
        step = (stop - start) * 1.0 / end_mipmap

        for i in range(start_mipmap, total_mipmap + 1):

            # frostbite, lagarde paper p67
            # http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr.pdf
            r = step * i
            # pow(r,1.5);
            roughness_linear = r

            size = int(math.pow(2, total_mipmap - i))
            filename = pattern_output.format(output, i)

            # generate debug color cubemap after limit size
            if i <= end_mipmap:
                print "compute level {} with roughness {} {} x {} to {}".format(
                    i, roughness_linear, size, size, filename)
                self.compute_level(i, size, roughness_linear, filename, num_samples, kwargs.get("fix_edge"))
            else:
                # write dummy cubemap
                face = numpy.empty((size, size), dtype=cl_array.vec.float4)
                face.fill((1.0, 0.0, 1.0, 0.0))
                cm = [face, face, face, face, face, face]
                write_image_cubemap(cm, filename)

    def d_create_image_cubemap(self, cubemap):
        """ create an image opencl from numpy image rgba """

        size = cubemap[0].shape[0]
        added_array = cubemap[0]
        for i in range(1, 6):
            added_array = numpy.append(added_array, cubemap[i])

        image = cl.Image(self.ctx,
                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                         cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                         (size, size, 6),
                         None,
                         added_array,
                         True)
        return image


def write_image_cubemap(cubemap, filename):

    output = oiio.ImageOutput.create(filename)
    mode = oiio.Create
    size = cubemap[0].shape[0]
    # print "size {}".format(cubemap[0].shape[0])
    spec_rgba = oiio.ImageSpec(size, size, 3, oiio.FLOAT)

    for face in cubemap:
        output.open(filename, spec_rgba, mode)
        # can't transform to rgb
        s = face.shape[0]
        # delete the A from RGBA
        f = face.view(dtype=numpy.float32).reshape(s * s, 4)
        f = numpy.delete(f, 3, 1)
        buffer = numpy.getbuffer(f)
        output.write_image(oiio.FLOAT, buffer)
        mode = oiio.AppendSubimage

    output.close()


def write_image(image, filename):

    output = oiio.ImageOutput.create(filename)
    size = image.shape[0]
    spec_rgba = oiio.ImageSpec(size, size, 4, oiio.FLOAT)
    output.open(filename, spec_rgba, oiio.Create)
    # can't transform to rgb
    output.write_image(oiio.FLOAT, numpy.getbuffer(image))
    output.close()


def read_image_cubemap(filename):
    image_array = []
    input = oiio.ImageInput.open(filename)
    spec = input.spec()
    width = spec.width
    height = spec.height
    print "reading {} - {}x{}x{}".format(filename, width, height, spec.nchannels)
    for sub in range(6):
        buf = oiio.ImageBuf(filename, sub, 0)
        RGBA = oiio.ImageBuf()
        oiio.ImageBufAlgo.channels(RGBA, buf, ("R", "G", "B", 1.0), ("R", "G", "B", "A"))
        pixels = RGBA.get_pixels(oiio.FLOAT)
        image = numpy.frombuffer(numpy.getbuffer(numpy.float32(pixels)), dtype=cl_array.vec.float4)
        image = image.reshape(width, height)
        image_array.append(image)

    if len(image_array) != 6:
        print "not all faces have been read"
    return image_array


def test_image():
    cubemap = read_image_cubemap("test2.tif")
    write_image_cubemap(cubemap, "out.tif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="cubemap environment [ cubemap.tif cubemap_%d.tif ]")
    parser.add_argument("output", help="output directory")
    parser.add_argument("--sampleFile", action="store", dest="sample_file",
                        help="use file to use for samples by roughness", default=None)
    parser.add_argument("--ggxSamples", action="store", dest="ggx_samples",
                        help="nb samples to compute environment 1 to 65536", default=4096)
    parser.add_argument("--size", action="store", dest="size",
                        help="cubemap size for prefiltered texture", default=256)
    parser.add_argument("--fixedge", action="store_true", help="fix edge for cubemap")
    parser.add_argument("--backgroundSamples", action="store", dest="background_samples",
                        help="nb samples to compute background 1 to 65536", default=4096)
    parser.add_argument("--backgroundSize", action="store", dest="background_size",
                        help="cubemap size for background texture", default=256)
    parser.add_argument("--backgroundBlur", action="store", dest="background_blur",
                        help="how to blur the background, it uses the same code of prefiltering", default=0.1)

    args = parser.parse_args()
    input_file = args.file
    output_directory = args.output

    print(args)

    process = Prefilter(input_file)

    process.run_background_blur(os.path.join(output_directory, "background.tif"),
                                size=int(args.background_size),
                                num_samples=int(args.background_samples),
                                radius=float(args.background_blur),
                                fix_edge=args.fixedge)

    process.run_ggx(output_directory,
                    size=int(args.size),
                    num_samples=int(args.ggx_samples),
                    fix_edge=args.fixedge,
                    limit_size=8,
                    sample_file=args.sample_file)
