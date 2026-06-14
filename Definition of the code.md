## First prototype:
The first working prototype should be simple and also no vibecoding.  
- TTS when a message is recieved on a specified channel the program should synthesize the text into voice and play it on a speaker.  
  - Piper
- STT when a button (PPT) is held the computer strarts recording a sample until the button is depressed {LOL}, some TBD algoritm is going to transalte the recording into text and it's going to send it into a specified chanel.  
  - IDK it's harder than it seemed
  - For now Whisper
  - then I'll think about FireRedASR

### Code cycle
- init
- press and hold `t` key it will record an audio clip
- use SST to transfer it into text
- ~~send to a known channel in MC~~
- Use TTS to speak it back out
- repeat from #2

## Ultimate goal
### Features
- Nice enclosure
- battery big enough to have atleast a day of runtime
  - Super low power consumption
- Physical PPT button
  - It beeps on listening and release and does a sound when it sends the message
- It should have reliable and natural TTS and STT
- A screen would be nice
- One main python code along with instalation script
- automagic language identification
- Speaker and a mic


## Resources
Maybe RTOS would be a good fit for this
- Also to note that the Raspberry Pi Zero 2 W supports I2S microphones.  
    https://learn.adafruit.com/adafruit-i2s-mems-microphone-breakout/raspberry-pi-wiring-test  
- https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#gpio  
- raspbian and dietpi are comparable in power usage.  
- https://github.com/k2-fsa/sherpa-onnx - All in one library
### TTS
- https://github.com/KittenML/KittenTTS - It sounds so good
### STT
- https://github.com/joycea17/vosk_kaldi - It's pretty terible, but there isn't anything better
- https://github.com/openai/whisper - It's super expensive on the CPU