import math
import copy
import numpy as np
import sys
import os
import json

def mirror_sequence(datasequence):
	new_datasequence = copy.deepcopy(datasequence)
	for data in new_datasequence['sequence']:
		# Robot
		data['robot']['y'] = -data['robot']['y']
		data['robot']['angle'] = -data['robot']['angle'] #math.atan2(math.sin(-data['robot']['angle']), math.cos(-data['robot']['angle']))
		data['robot']['speed_y'] = -data['robot']['speed_y']
		data['robot']['speed_a'] = -data['robot']['speed_a']
  
		# People
		for i in range(len(data['people'])):
			data['people'][i]['y'] = -data['people'][i]['y']
			data['people'][i]['angle'] = -data['people'][i]['angle']#math.atan2(math.sin(-data['people'][i]['angle']), math.cos(-data['people'][i]['angle']))
   
		# Objects
		for i in range(len(data['objects'])):
			data['objects'][i]['y'] = -data['objects'][i]['y']
			data['objects'][i]['angle'] = -data['objects'][i]['angle']#math.atan2(math.sin(-data['objects'][i]['angle']), math.cos(-data['objects'][i]['angle']))

		# Goal
		data['goal']['y'] = -data['goal']['y']
		data['goal']['angle'] = -data['goal']['angle'] #math.atan2(math.sin(-data['goal']['angle']), math.cos(-data['goal']['angle']))

	for i in range(len(new_datasequence['walls'])):
		new_datasequence['walls'][i][0], new_datasequence['walls'][i][2] = new_datasequence['walls'][i][2], new_datasequence['walls'][i][0]
		new_datasequence['walls'][i][1], new_datasequence['walls'][i][3] = -new_datasequence['walls'][i][3], -new_datasequence['walls'][i][1]


	return new_datasequence


