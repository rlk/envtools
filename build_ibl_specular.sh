
src="$1"
dst="$2"

image_magick=""

iconvert "${src}" /tmp/base.tif
./envremap -o cube /tmp/base.tif /tmp/cubemap.tif
./envtospecular /tmp/cubemap.tif /tmp/specular

offset=0
for size in 1024 512 256 128 64 32 16 8 4 2
do
    base="/tmp/specular"
    let "size1=${size}*2"
    filename="$(ls ${base}_${size}_*.tif)"
    if [ $? -ne 0 ]
    then
        continue
    fi

    let "size2=${size1}*2"

    filename_output="${base}_rect_${size1}.tif"
    ./envremap -i cube -o rect -n "${size1}" "${filename}" "${filename_output}"
    colorspace="-colorspace rgb"
    #colorspace=""
    if [ -z "${image_magick}" ]
    then
        image_magick="convert -page ${size2}x${size2}+0+0 ${filename_output} ${colorspace}"
    else
        image_magick="${image_magick} -page +0+${offset} ${filename_output} ${colorspace}"
    fi
    let "offset+=${size1}"
done
$(${image_magick} -flatten ${dst})
