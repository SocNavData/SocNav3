for f in `find . -name *.mp4 -type f`
do
    ffmpeg -ss 00:00:00.5 -i ${f} -vcodec libx264 -crf 28 -threads 10 /tmp/sn3_output.mp4
    mv /tmp/sn3_output.mp4 ${f}
done

