
base_name="${1}"
dst="${2}"

image_magick=""

offset=0
for current_size in 2048 1024 512 256 128 64 32 16 8 4 2 1
do
    base="${base_name}"
    let "sizey=${current_size}/2"
    filename="$(ls ${base}${current_size}.tif 2>/dev/null)"
    if [ "$?" -ne 0 ]
    then
        continue
    fi

    colorspace="-colorspace rgb"
    if [ -z "${image_magick}" ]
    then
        image_magick="convert -page ${current_size}x${current_size}+0+0 ${filename} ${colorspace}"
    else
        image_magick="${image_magick} -page +0+${offset} ${filename} ${colorspace}"
    fi
    let "offset+=${sizey}"
done
$(${image_magick} -flatten ${dst})
