# LauraGPT TSE Only Clean


This is the code for trying to padding the input as 

SOS ref EMB mix SOS clean SOS

Note that "_new" means the new version which fixed the dynamic mixing bug where the interferring speech is only clipped using the first 5 secs rather than randomly chosen from the clip.


## v1 update

- reference speech 
    - length can be controlled within a range. I set it 5-10 seconds. 
    - any part of the speech can work rather than just the last 5 seconds.

- data dynamic mixing
    - Add scripts to combine data

## TODO:
- [x] add `fine_tune` to the recipes sh