#!/usr/bin/python
import subprocess
import sys
import os
import math
import json

epsilon = 0.01

output_directory = "out"
test_data_directory = "testData"
output_json = "outTest.json"
extract_lights = "extractLights"

error_message = "[ERROR] extractlights output diff changed"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)    
if os.path.exists(os.path.join(test_data_directory, "debug_variance.png")):
    os.remove(os.path.join(test_data_directory, "debug_variance.png"))   
if os.path.exists(output_json):
    os.remove(output_json)

input_panorama = os.path.join(test_data_directory, "graceCath.jpeg")
cmd = "{} -d {} > {}".format(os.path.join("./",extract_lights), input_panorama, output_json)
output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

original_data = json.load(open(os.path.join(test_data_directory, "outTest.json")))
test_data = json.load(open("outTest.json"))

if len(test_data) != len(original_data):
    sys.exit(error_message)

k = 0
for i in test_data:
    lum_test = float(i['lum_ratio'])
    lum_orig = float(original_data[k]['lum_ratio'])
    if (math.fabs(lum_test - lum_orig) > 0.001):
        sys.exit(error_message)
    k += 1

    
print "[OK] extractLights output diff test OK"
