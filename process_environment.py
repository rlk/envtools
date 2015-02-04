#!/usr/bin/python

import subprocess
import sys
import os
import time
import math
import json
import glob
import argparse

DEBUG=False

envIrradiance_cmd="envIrradiance"
envPrefilter_cmd="envPrefilter"
envIntegrateBRDF_cmd="envBRDF"
cubemap_packer_cmd="cubemapPacker"
panorama_packer_cmd="panoramaPacker"
envremap_cmd="envremap"
seamlessCubemap_cmd="seamlessCubemap"
envBackground_cmd="envBackground"
compress_7Zip_cmd="7z"

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
        self.input_file = os.path.abspath(input_file)
        self.output_directory = output_directory
        self.pretty = kwargs.get("pretty", False )
        self.integrate_BRDF_size = kwargs.get("brdf_texture_size", 128 )
        self.irradiance_size = kwargs.get("irradiance_size", 32 )
        self.pattern_filter = kwargs.get("pattern_filter", "rgss" )
        self.nb_samples = kwargs.get("nb_samples", "1024" )
        self.prefilter_stop_size = kwargs.get("prefilter_stop_size", 8 )
        self.fixedge = kwargs.get("fixedge", False )

        self.specular_size = kwargs.get("specular_size", 512 )
        self.specular_file_base = "specular"

        self.brdf_file = 'brdf_ue4.bin'

        self.background_size = kwargs.get("background_size", 256 )
        self.background_blur = kwargs.get("background_blur", 0.1 )
        self.background_file_base = 'background_cubemap'

        self.mipmap_file_base = 'mipmap_cubemap'

        self.encoding_list = [ "rgbm", "luv", "float" ]
        self.can_comppress = True if which(compress_7Zip_cmd) != None else False


    def writeConfig(self, filename):
        output = open( filename, "w" )

        compress_extension = '.gz' if self.can_comppress is True else ''

        config = {

            "backgroundBlur": self.background_blur,
            "backgroundCubemapSize": [ self.background_size, self.background_size ],

            "mipmapCubemapSize": [ self.specular_size, self.specular_size ],
            "specularCubemapUE4Size": [ self.specular_size, self.specular_size ],
            "specularPanoramaUE4Size": [ self.specular_size * 4, self.specular_size * 4 ],
            "specularLimitSize": self.prefilter_stop_size,

            "brdfUE4": self.brdf_file + compress_extension,
            "brdfUE4Size": [ self.integrate_BRDF_size, self.integrate_BRDF_size ],

            "diffuseSPH": json.loads(self.sh_coef)
        }

        for enc in self.encoding_list:
            file = "{}_{}.bin{}".format( self.background_file_base, enc, compress_extension)
            key = "backgroundCubemap_{}".format( enc )
            config[key] = file

            for env_type in [ "Panorama", "Cubemap" ]:
                file = "{}_{}_ue4_{}.bin{}".format( self.specular_file_base, env_type.lower(), enc, compress_extension)
                key = "specular{}UE4_{}".format( env_type, enc )
                config[key] = file

            key = "mipmapCubemap_{}".format( enc )
            config[key] = "{}_{}.bin{}".format(self.mipmap_file_base, enc, compress_extension)


        if self.pretty is True:
            json.dump(config, output , sort_keys=True, indent=4)
        else:
            json.dump(config, output)

    def compress(self):
        files = glob.glob( os.path.join(self.output_directory, '*.bin') )
        for f in files:
            cmd = "{} a -tgzip -mx=9 -mpass=7 {}.gz {}".format(compress_7Zip_cmd, f, f)
            output = execute_command(cmd, verbose=False)
            os.remove( f )

    def encode_texture(self, input_file, output_directory=None):

        if not output_directory:
            output_directory = os.path.dirname(input_file)

        filename = os.path.basename(input_file)
        filename_without_extension = os.path.splitext(filename)[0]

        output = os.path.join(output_directory, filename_without_extension) + ".png"

        cmd = "{} -m rgbe {} {}".format(rgbx_cmd, input_file, output)
        output = execute_command(cmd, verbose=False)

    def extract_cubemap_face_and_encode(self, input, output, index):

        output_file = "{}_{}.tif".format(output, index)
        cmd = "oiiotool {} -subimage {} -o {}".format(input, index, output_file)
        execute_command(cmd)
        self.encode_texture(output_file)

    def create_cubemap(self, input_cubemap, output_directory):

        for cubemap_face in range(0, 6):
            self.extract_cubemap_face_and_encode(input_cubemap, output_directory, cubemap_face)

    def compute_irradiance(self, input):

        tmp = "/tmp/irr.tif"

        cmd = "{} -n {} {} {}".format( envIrradiance_cmd, self.irradiance_size, input, tmp)
        output_log = execute_command(cmd, verbose=False, print_command=True)

        lines_list = output_log.split("\n")
        for line in lines_list:
            index = line.find("shCoef:")
            if index != -1:
                self.sh_coef = line[line.find(":") + 1:]
                # with open( os.path.join(self.output_directory, "spherical"), "w") as f:
                #     f.write(self.sh_coef)
                # break


        # generate texture for irrandiance
        # since we use spherical harmonics we dont need textures anymore except for debugging
        if DEBUG is True:
            self.create_cubemap(tmp, os.path.join(self.output_directory, "cubemap_irradiance"))

            # compute the panorama version of irradiance
            panorama_size = self.irradiance_size * 2
            panorama_irradiance = "/tmp/panorama_irradiance.tif"

            cmd = "{} -n {} -i cube -o rect {} {}".format( envremap_cmd, panorama_size, tmp, panorama_irradiance)
            execute_command(cmd)

            self.encode_texture(panorama_irradiance, self.output_directory)

    def cubemap_fix_border(self, input, output):
        cmd = "{} {} {}".format(seamlessCubemap_cmd, input, output)
        execute_command(cmd)

    def cubemap_packer(self, pattern, max_level, output ):
        cmd = ""
        if max_level > 0:
            cmd = "{} -p -n {} {} {}".format(cubemap_packer_cmd, max_level, pattern, output)
        else:
            cmd = "{} {} {}".format(cubemap_packer_cmd, pattern, output)
        execute_command(cmd)

    def panorama_packer(self, pattern, max_level, output ):
        cmd = "{} {} {} {}".format(panorama_packer_cmd, pattern, max_level, output)
        execute_command(cmd)

    def getMaxLevel(self, value ):
        max_level = int(math.log(float(self.specular_size)) / math.log(2))
        return max_level

    def cubemap_specular_create_mipmap(self, input):

        max_level = self.getMaxLevel(self.specular_size)

        previous_file = self.cubemap_generic
        self.cubemap_fix_border(previous_file, "/tmp/fixup_0.tif")

        for i in range(1, max_level + 1):
            size = int(math.pow(2, max_level - i))
            level = i;
            outout_filename = "/tmp/specular_{}.tif".format(i)
            cmd = "{} -p {} -n {} -i cube -o cube {} {}".format(envremap_cmd, self.pattern_filter, size, previous_file, outout_filename)
            previous_file = outout_filename
            execute_command(cmd)
            self.cubemap_fix_border(outout_filename, "/tmp/fixup_{}.tif".format( i ) )

        self.cubemap_packer("/tmp/fixup_%d.tif", max_level, os.path.join(self.output_directory, self.mipmap_file_base) )



    def cubemap_specular_create_prefilter(self, input ):
        import shutil

        max_level = self.getMaxLevel(self.specular_size)

        # compute it one time for panorama
        outout_filename = "/tmp/prefilter_specular"
        cmd = "{} -s {} -e {} -n {} {} {}".format(envPrefilter_cmd, self.specular_size, self.prefilter_stop_size, self.nb_samples, self.cubemap_generic , outout_filename)
        execute_command(cmd)

        # compute it seamless for cubemap
        outout_filename = "/tmp/prefilter_fixup"
        cmd = "{} -s {} -e {} -n {} -f {} {}".format(envPrefilter_cmd, self.specular_size, self.prefilter_stop_size, self.nb_samples, self.cubemap_generic , outout_filename)
        execute_command(cmd)

        self.cubemap_packer("/tmp/prefilter_fixup_%d.tif", max_level, os.path.join(self.output_directory, "specular_cubemap_ue4"))

        # create the integrateBRDF texture
        # we dont need to recreate it each time
        outout_filename = os.path.join(self.output_directory, "brdf_ue4.bin" )
        cmd = "{} -s {} -n {} {}".format(envIntegrateBRDF_cmd, self.integrate_BRDF_size, self.nb_samples, outout_filename)
        execute_command(cmd)


    def background_create( self ):

        # compute it one time for panorama
        outout_filename = "/tmp/background.tiff"
        fixedge = '-f' if self.fixedge else ''
        cmd = "{} -s {} -n {} -r {} {} {} {}".format(envBackground_cmd, self.background_size, self.nb_samples, self.background_blur , fixedge, self.cubemap_generic, outout_filename)
        execute_command(cmd)

        # packer use a pattern, fix cubemap packer ?
        input_filename = outout_filename
        output = os.path.join(self.output_directory, self.background_file_base )
        self.cubemap_packer( input_filename, 0, output )


    # use the step from cubemap prefilter
    def panorama_specular_create_prefilter(self):

        # panorama is 4 * cubemap face
        # cubemap of 512 -> panorama 2048

        # but we dont change the generation of lod, we will not use
        # end of mipmap level
        max_cubemap_level = int(math.log(self.specular_size) / math.log(2)) + 1
        max_level = int(math.log(self.specular_size*4) / math.log(2)) + 1

        for i in range(1, max_cubemap_level):
            level = i - 1
            size = pow(2, max_level - i )
            input_filename = "/tmp/prefilter_specular_{}.tif".format(level)
            output_filename = "/tmp/panorama_prefilter_specular_{}.tif".format(level)
            cmd = "{} -p {} -n {} -i cube -o rect {} {}".format(envremap_cmd, self.pattern_filter, size/2, input_filename, output_filename)
            execute_command(cmd)

        self.panorama_packer("/tmp/panorama_prefilter_specular_%d.tif", max_level - 1, os.path.join(self.output_directory, "specular_panorama_ue4"))

    def panorama_specular(self, input):

        # compute the panorama from cubemap specular
        panorama_size = self.specular_size * 2
        panorama_specular = "/tmp/panorama.tif"
        cmd = "{} -p {} -n {} -i rect -o rect {} {}".format( envremap_cmd, self.pattern_filter, panorama_size, input, panorama_specular)
        execute_command(cmd)

        self.encode_texture(panorama_specular, self.output_directory)

    def run(self):

        start = time.time()

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        cubemap_generic = "/tmp/input_cubemap.tif"

        original_file = "/tmp/original_panorama.tif"
        cmd = "iconvert {} {}".format(self.input_file, original_file )
        execute_command(cmd)

        cmd = "{} -p {} -o cube -n {} {} {}".format(envremap_cmd, self.pattern_filter, self.specular_size, original_file, cubemap_generic)
        self.cubemap_generic = cubemap_generic
        execute_command(cmd)

        # create cubemap original for debug
        if DEBUG is True:
            self.create_cubemap(cubemap_generic, os.path.join(self.output_directory, 'cubemap'))

        # generate irradiance*PI panorama/cubemap/sph
        self.compute_irradiance(cubemap_generic)

        # generate specular
        self.cubemap_specular_create_mipmap(cubemap_generic)

        # generate panorama specular
        #self.panorama_specular(original_file)


        # generate prefilter ue4 specular
        self.cubemap_specular_create_prefilter(cubemap_generic)

        # generate prefilter ue4 specular panorama
        self.panorama_specular_create_prefilter()


        # generate background
        self.background_create()


        # write config for this environment
        self.writeConfig( os.path.join(self.output_directory, "config.json" ) )


        if self.can_comppress:
            self.compress()

        print ("processed in {} seconds".format(time.time() - start))


parser = argparse.ArgumentParser()
parser.add_argument("file", help="hdr environment [ .hdr .tif .exr ]")
parser.add_argument("output", help="output directory")
parser.add_argument("--nbSamples", action='store', dest='nb_samples', help="nb samples to compute environment 1 to 65536", default=65536)
parser.add_argument("--specularSize", action='store', dest='specular_size', help="cubemap size for prefiltered texture", default=256)
parser.add_argument("--backgroundSize", action='store', dest='background_size', help="cubemap size for background texture", default=256)
parser.add_argument("--backgroundBlur", action='store', dest='background_blur', help="how to blur the background, it uses the same code of prefiltering", default=0.1)
parser.add_argument("--fixedge", action='store_true', help="fix edge for cubemap")
parser.add_argument("--pretty", action='store_true', help="generate a config file pretty for human")



args = parser.parse_args()
input_file = args.file
output_directory = args.output

print(args)
process = ProcessEnvironment( input_file,
                              output_directory,
                              background_size = args.background_size,
                              specular_size = args.specular_size,
                              nb_samples = args.nb_samples,
                              background_blur = args.background_blur,
                              prefilter_stop_size = 8,
                              fixedge = args.fixedge,
                              pretty = args.pretty )
process.run()
