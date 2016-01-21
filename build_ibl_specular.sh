src="$1"
dst="$2"
size="$3"

pid=$$

# you need to have openimageio
iinfo --stats "${src}" | grep '6 subimages'
already_cubemap=$?

if [ ! ${already_cubemap} ]
then
    iconvert "${src}" /tmp/base_${pid}.tif

    let "cubemap_size=${size}/2"

    ./envremap -o cube -n $cubemap_size /tmp/base_${pid}.tif /tmp/cubemap_${pid}.tif
    base_name=/tmp/specular_${pid}

    src="/tmp/cubemap_${pid}.tif"
fi

./envtospecular "${src}" "${base_name}"


for current_size in 1024 512 256 128 64 32 16 8 4 2 1
do
    base="${base_name}"
    let "width_rect=${current_size}*2"
    filename="$(ls ${base}_${current_size}_*.tif 2>/dev/null)"
    if [ "$?" -ne 0 ]
    then
        continue
    fi

    filename_output="${base}_rect_${width_rect}.tif"
    ./envremap -i cube -o rect -n "${current_size}" "${filename}" "${filename_output}"

done

bash -x ./build_mipmap.sh "${base}_rect_" "${dst}"
