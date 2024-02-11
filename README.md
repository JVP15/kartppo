

## Installation

On linux, I ran into an issue running the code when using Conda. The solution was to follow this stack overflow post:
https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris, and also the comment below targetting the specific conda envronment

So I did something like

```bash
cd /home/$USER/miniconda3/$ENV/lib
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ # Copy the c++ dynamic link library of the system here
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.6.0.19
```