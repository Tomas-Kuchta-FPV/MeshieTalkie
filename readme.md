# Meshie Talkie
> Like a walkie talkie but Mesh Core!  
>![banner](/AIgen%20banner.png)  

So this code aimes to make low power walkie talkie experience possible with MeshCore.  

*How does it work?*  
TX: You press and hold the PTT key, it saves the recording into /tmp and after key release it transcribes it into text whic then sends to a predefined MC channel.  
RX: When a mesage gets to your device via a certain channel the PPT model synthesizes it and saves it as an audio file which then get's played through the speakers.  
## Instalation instructions
`cd src`  
`pip install -r requirements.txt`  
`python main.py`  
