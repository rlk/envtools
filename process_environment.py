#!/usr/bin/python

import subprocess
import sys
import os
import time
import math
import json
import argparse

DEBUG = False

envIrradiance_cmd = "envIrradiance"
envPrefilter_cmd = "envPrefilter"
envIntegrateBRDF_cmd = "envBRDF"
cubemap_packer_cmd = "cubemapPacker"
panorama_packer_cmd = "panoramaPacker"
envremap_cmd = "envremap"
seamlessCubemap_cmd = "seamlessCubemap"
envBackground_cmd = "envBackground"
compress_7Zip_cmd = "7z"


def execute_command(cmd, **kwargs):

    try:
        start = 0
        end = 0
        if kwargs.get("profile", True):
            start = time.time()

        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

        if kwargs.get("profile", True):
            end = time.time()

        if kwargs.get("verbose", True) or kwargs.get("print_command", False):
            print ("{} - {}".format(end - start, cmd))

        if kwargs.get("verbose", True) and output:
            print (output)

        return output

    except subprocess.CalledProcessError as error:
        print("error {} executing {}".format(error.returncode, cmd))
        print(error.output)
        sys.exit(1)
        return None


def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


class ProcessEnvironment(object):

    def __init__(self, input_file, output_directory, **kwargs):
        self.encoding_type = ["luv", "rgbm", "rgbe", "float"]
        self.input_file = os.path.abspath(input_file)
        self.output_directory = output_directory
        self.pretty = kwargs.get("pretty", False)

        self.process_only_bg = kwargs.get("process_only_background", False)

        self.integrate_BRDF_size = kwargs.get("brdf_texture_size", 128)
        self.irradiance_size = kwargs.get("irradiance_size", 32)
        self.pattern_filter = kwargs.get("pattern_filter", "rgss")
        self.nb_samples = kwargs.get("nb_samples", "1024")
        self.background_samples = kwargs.get("background_samples", "1024")
        self.prefilter_stop_size = kwargs.get("prefilter_stop_size", 8)
        self.thumbnail_size = kwargs.get("thumbnail_size", 256)
        self.fixedge = kwargs.get("fixedge", False)
        self.write_by_channel = kwargs.get("write_by_channel", False)

        self.specular_size = kwargs.get("specular_size", 512)
        self.specular_file_base = "specular"

        self.brdf_file = "brdf_ue4.bin"

        self.background_size = kwargs.get("background_size", 256)
        self.background_blur = kwargs.get("background_blur", 0.1)
        self.background_file_base = "background"

        self.mipmap_file_base = "mipmap_cubemap"

        self.can_comppress = True if which(compress_7Zip_cmd) != None else False

        self.config = {'textures': []}
        self.textures = {}

    def writeConfig(self):
        filename = os.path.join(self.output_directory, "config.json")
        output = open(filename, "w")

        config = self.config
        config["diffuseSPH"] = json.loads(self.sh_coef)

        for texture in self.config['textures']:
            for image in texture['images']:
                f = image["file"]
                image["file"] = os.path.relpath(f, self.output_directory)

        if self.pretty is True:
            json.dump(config, output, sort_keys=True, indent=4)
        else:
            json.dump(config, output)

    def compress(self):
        for texture in self.config['textures']:
            if texture['type'] == 'thumbnail':
                continue
            for image in texture['images']:
                f = image["file"]
                size_before = os.path.getsize(f)
                cmd = "{} a -tgzip -mx=9 -mpass=7 {}.gz {}".format(compress_7Zip_cmd, f, f)
                execute_command(cmd, verbose=False)
                size_after = os.path.getsize(f + '.gz')
                image["file"] = "{}.gz".format(f)
                image["sizeUncompressed"] = size_before
                image["sizeCompressed"] = size_after
                os.remove(f)

    def compute_irradiance(self):

        tmp = "/tmp/irr.tif"

        cmd = "{} -n {} {} {}".format(envIrradiance_cmd, self.irradiance_size, self.cubemap_highres, tmp)
        output_log = execute_command(cmd, verbose=False, print_command=True)

        lines_list = output_log.split("\n")
        for line in lines_list:
            index = line.find("shCoef:")
            if index != -1:
                self.sh_coef = line[line.find(":") + 1:]
                # with open( os.path.join(self.output_directory, "spherical"), "w") as f:
                #     f.write(self.sh_coef)
                # break

    def cubemap_fix_border(self, input, output):
        cmd = "{} {} {}".format(seamlessCubemap_cmd, input, output)
        execute_command(cmd)

    def cubemap_packer(self, pattern, max_level, output):
        cmd = ""
        write_by_channel = "-c" if self.write_by_channel else ""
        if max_level > 0:
            cmd = "{} {} -p -n {} {} {}".format(cubemap_packer_cmd, write_by_channel, max_level, pattern, output)
        else:
            cmd = "{} {} {} {}".format(cubemap_packer_cmd, write_by_channel, pattern, output)
        execute_command(cmd)

    def panorama_packer(self, pattern, max_level, output):
        write_by_channel = "-c" if self.write_by_channel else ""
        cmd = "{} {} {} {} {}".format(panorama_packer_cmd, write_by_channel, pattern, max_level, output)
        execute_command(cmd)

    def getMaxLevel(self, value):
        max_level = int(math.log(float(value)) / math.log(2))
        return max_level

    def cubemap_specular_create_mipmap(self, specular_size):

        max_level = self.getMaxLevel(specular_size)

        previous_file = self.cubemap_highres

        for i in range(0, max_level + 1):
            size = int(math.pow(2, max_level - i))
            outout_filename = "/tmp/specular_{}.tif".format(i)
            cmd = "{} -p {} -n {} -i cube -o cube {} {}".format(
                envremap_cmd, self.pattern_filter, size,
                previous_file, outout_filename)

            previous_file = outout_filename
            execute_command(cmd)
            self.cubemap_fix_border(outout_filename, "/tmp/fixup_{}.tif".format(i))

        file_basename = os.path.join(self.output_directory, self.mipmap_file_base)
        self.cubemap_packer("/tmp/fixup_%d.tif", max_level, file_basename)

        for encoding in self.encoding_type:
            file_to_check = "{}_{}.bin".format(file_basename, encoding)
            if os.path.exists(file_to_check) is True:
                self.registerImageConfig(encoding, "cubemap", "mipmap", 8, {
                    "width": specular_size,
                    "height": specular_size,
                    "file": file_to_check
                })

    def compute_brdf_lut_ue4(self):
        # create the integrateBRDF texture
        # we dont need to recreate it each time
        outout_filename = os.path.join(self.output_directory, "brdf_ue4.bin")
        size = self.integrate_BRDF_size
        cmd = "{} -s {} -n {} {}".format(envIntegrateBRDF_cmd, size, self.nb_samples, outout_filename)
        execute_command(cmd)

        self.registerImageConfig('rg16', 'lut', "brdf_ue4", None, {
            "width": size,
            "height": size,
            "file": outout_filename,
            "samples": self.nb_samples
        })

    def registerImageConfig(self, image_encoding, image_format, image_type, limitSize, config):
        key = image_encoding + image_format + image_type
        if 'nbSamples' in config:
            key += str(config['nbSamples'])

        entry = None
        if key not in self.textures:
            entry = {
                'type': image_type,
                'format': image_format,
                'encoding': image_encoding,
                'images': []
            }
            if limitSize is not None:
                entry['limitSize'] = limitSize

            self.config['textures'].append(entry)
            self.textures[key] = entry
        else:
            entry = self.textures[key]
        entry['images'].append(config)

    def process_cubemap_specular_create_prefilter(self, specular_size, prefilter_stop_size, fixedge, output_filename):
        fix_flag = "-f" if fixedge else ""

        cmd = "{} -s {} -e {} -n {} {} {} {}".format(
            envPrefilter_cmd, specular_size, prefilter_stop_size,
            self.nb_samples, fix_flag, self.cubemap_highres,
            output_filename)
        execute_command(cmd)

    def specular_create_prefilter_panorama(self, specular_size, prefilter_stop_size):
        max_level = self.getMaxLevel(specular_size)

        tmp_filename = "/tmp/prefilter_specular"
        self.process_cubemap_specular_create_prefilter(specular_size, prefilter_stop_size, False, tmp_filename)

        panorama_size = specular_size * 4

        # panorama is 4 * cubemap face
        # cubemap of 512 -> panorama 2048

        # but we dont change the generation of lod, we will not use
        # end of mipmap level
        max_cubemap_level = self.getMaxLevel(specular_size) + 1
        max_level = self.getMaxLevel(panorama_size) + 1
        for i in range(1, max_cubemap_level):
            level = i - 1
            size = pow(2, max_level - i)
            input_filename = "{}_{}.tif".format(tmp_filename, level)
            output_filename = "/tmp/panorama_prefilter_specular_{}.tif".format(level)
            cmd = "{} -p {} -n {} -i cube -o rect {} {}".format(
                envremap_cmd, self.pattern_filter, size / 2,
                input_filename, output_filename)
            execute_command(cmd)

        file_basename = os.path.join(self.output_directory, "specular_panorama_ue4_{}".format(panorama_size))

        self.panorama_packer("/tmp/panorama_prefilter_specular_%d.tif", max_level - 1,
                             file_basename)

        for encoding in self.encoding_type:
            file_to_check = "{}_{}.bin".format(file_basename, encoding)
            if os.path.exists(file_to_check) is True:
                self.registerImageConfig(encoding, "panorama", "specular_ue4", prefilter_stop_size * 4, {
                    "width": panorama_size,
                    "height": panorama_size,
                    "file": file_to_check,
                    "samples": self.nb_samples
                })

    def specular_create_prefilter_cubemap(self, specular_size, prefilter_stop_size):

        max_level = self.getMaxLevel(specular_size)
        self.process_cubemap_specular_create_prefilter(
            specular_size, prefilter_stop_size, True, "/tmp/prefilter_fixup")

        file_basename = os.path.join(self.output_directory, "specular_cubemap_ue4_{}".format(specular_size))
        self.cubemap_packer(
            "/tmp/prefilter_fixup_%d.tif", max_level, file_basename)

        for encoding in self.encoding_type:
            file_to_check = "{}_{}.bin".format(file_basename, encoding)
            if os.path.exists(file_to_check) is True:
                self.registerImageConfig(encoding, "cubemap", "specular_ue4", prefilter_stop_size, {
                    "width": specular_size,
                    "height": specular_size,
                    "file": file_to_check,
                    "samples": self.nb_samples
                })

    def specular_create_prefilter(self, specular_size, prefilter_stop_size):
        self.specular_create_prefilter_panorama(specular_size, prefilter_stop_size)
        self.specular_create_prefilter_cubemap(specular_size, prefilter_stop_size)

    def background_create(self, background_size, background_blur, background_samples=None):

        samples = self.background_samples
        if background_samples is not None:
            samples = background_samples

        # compute it one time for panorama
        output_filename = "/tmp/background.tiff"
        fixedge = "-f" if self.fixedge else ""
        cmd = "{} -s {} -n {} -r {} {} {} {}".format(
            envBackground_cmd, background_size, samples,
            background_blur, fixedge, self.cubemap_highres,
            output_filename)

        execute_command(cmd)

        # packer use a pattern, fix cubemap packer ?
        file_basename = os.path.join(self.output_directory, "{}_cubemap_{}_{}".format(
            self.background_file_base,
            background_size,
            background_blur))
        self.cubemap_packer(output_filename, 0, file_basename)

        for encoding in self.encoding_type:
            file_to_check = "{}_{}.bin".format(file_basename, encoding)
            if os.path.exists(file_to_check) is True:
                self.registerImageConfig(encoding, "cubemap", "background", None, {
                    "width": background_size,
                    "height": background_size,
                    "blur": background_blur,
                    "file": file_to_check,
                    "samples": samples
                })

    def thumbnail_create(self, thumbnail_size):

        # compute it one time for panorama
        file_basename = os.path.join(self.output_directory, "thumbnail_{}.jpg".format(thumbnail_size))
        cmd = "oiiotool {} --resize {}x{} --cpow 0.45454545,0.45454545,0.45454545,1.0 -o {}".format(
            self.panorama_highres, thumbnail_size, thumbnail_size / 2, file_basename)
        execute_command(cmd)

        self.registerImageConfig("srgb", "panorama", "thumbnail", None, {
            "width": thumbnail_size,
            "height": thumbnail_size / 2,
            "file": file_basename
        })

    def initBaseTexture(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        original_file = "/tmp/original_panorama.tif"
        cmd = "iconvert '{}' {}".format(self.input_file, original_file)
        execute_command(cmd)
        self.panorama_highres = original_file

        cubemap_highres = "/tmp/highres_cubemap.tif"
        cmd = "{} -p {} -o cube {} {}".format(envremap_cmd, self.pattern_filter, original_file, cubemap_highres)
        execute_command(cmd)

        self.cubemap_highres = cubemap_highres

    def run(self):

        start = time.time()

        self.initBaseTexture()

        # generate thumbnail
        self.thumbnail_create(self.thumbnail_size)

        # generate background
        self.background_create(self.background_size, self.background_blur)

        if self.process_only_bg:
            if self.can_comppress:
                self.compress()
            return

        # generate irradiance*PI panorama/cubemap/sph
        self.compute_irradiance()

        # generate specular
        self.cubemap_specular_create_mipmap(self.specular_size)

        # precompute lut brdf
        self.compute_brdf_lut_ue4()

        # generate prefilter ue4 specular
        self.specular_create_prefilter(self.specular_size, self.prefilter_stop_size)

        if self.can_comppress:
            self.compress()

        # write config for this environment
        self.writeConfig()

        print ("processed in {} seconds".format(time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="hdr environment [ .hdr .tif .exr ]")
    parser.add_argument("output", help="output directory")
    parser.add_argument("--write-by-channel", action="store_true", dest="write_by_channel",
                        help="write by channel all red then green then blue ...")
    parser.add_argument("--nbSamples", action="store", dest="nb_samples",
                        help="nb samples to compute environment 1 to 65536", default=65536)
    parser.add_argument("--backgroundSamples", action="store", dest="background_samples",
                        help="nb samples to compute background 1 to 65536", default=4096)
    parser.add_argument("--specularSize", action="store", dest="specular_size",
                        help="cubemap size for prefiltered texture", default=256)
    parser.add_argument("--thumbnailSize", action="store", dest="thumbnail_size",
                        help="cubemap size for prefiltered texture", default=256),
    parser.add_argument("--backgroundSize", action="store", dest="background_size",
                        help="cubemap size for background texture", default=256)
    parser.add_argument("--backgroundBlur", action="store", dest="background_blur",
                        help="how to blur the background, it uses the same code of prefiltering", default=0.1)
    parser.add_argument("--bgonly", action="store_true", help="process only background")
    parser.add_argument("--fixedge", action="store_true", help="fix edge for cubemap")
    parser.add_argument("--pretty", action="store_true", help="generate a config file pretty for human")

    args = parser.parse_args()
    input_file = args.file
    output_directory = args.output

    print(args)
    process = ProcessEnvironment(input_file,
                                 output_directory,
                                 process_only_background=args.bgonly,
                                 thumbnail_size=int(args.thumbnail_size),
                                 background_samples=int(args.background_samples),
                                 background_size=int(args.background_size),
                                 specular_size=int(args.specular_size),
                                 nb_samples=int(args.nb_samples),
                                 background_blur=float(args.background_blur),
                                 write_by_channel=args.write_by_channel,
                                 prefilter_stop_size=8,
                                 fixedge=args.fixedge,
                                 pretty=args.pretty)
    process.run()
