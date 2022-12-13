ffmpeg -framerate 10 -i pca%04d.png -c:v libvpx -auto-alt-ref 0 out.webm
