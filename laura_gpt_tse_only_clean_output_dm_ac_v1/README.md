# LauraGPT TSE Only Clean


This is the code for trying to padding the input as 

SOS ref EMB mix SOS clean SOS


## v1 update

- reference speech 
    - length can be controlled within a range. I set it 5-10 seconds. 
    - any part of the speech can work rather than just the last 5 seconds.

- Update `MaxLength` class
    - Add a random clip option. If enabled, any random part will be chosen if given audio's length is greater than max length. Previously, only the first max_len will be chosen.


## TODO:
- [x] add `fine_tune` to the recipes sh