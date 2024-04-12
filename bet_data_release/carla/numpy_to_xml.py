import numpy as np
import matplotlib.pyplot as plt

Town10_spawns = np.load("bet_data_release/carla/Town10_spawns.npy")
Town04_spawns = np.load("bet_data_release/carla/Town04_spawns.npy")

headerXML = '''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <route id="1">
        <ego_vehicle id="hero">
'''

elements = ""

i = 0
for element in Town10_spawns:
    elements = elements+ '<waypoint pitch="%s" roll="%s" x="%s" y="%s" yaw="%s" z="%s" />\n'% (element[3],element[5],element[0],element[1],element[4],element[5])
    i = i + 1


footerXML = '''
        </ego_vehicle>
    </route>
</routes>
'''

XML =(headerXML+elements+footerXML)

outFile = open("bet_data_release/carla/Town10_spawns_coordinates.xml","w")
outFile.write(XML)
outFile.close()