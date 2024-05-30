from pathlib import Path
import google.generativeai as genai
from tqdm import tqdm
import os
import sys

genai.configure(api_key=sys.argv[1],transport="rest")

# Set up the model
generation_config = {
  "temperature": 0.4,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

commands = [line.strip() for line in open('command.txt')]

model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
example_images = [
    # "379a0ceca19e4e5ead2b5198f592febb_00.jpg",  
    # "379a0ceca19e4e5ead2b5198f592febb_01.jpg",  
    # "379a0ceca19e4e5ead2b5198f592febb_03.jpg",  
    # "379a0ceca19e4e5ead2b5198f592febb_04.jpg",  
    # "379a0ceca19e4e5ead2b5198f592febb_06.jpg",  
    # "379a0ceca19e4e5ead2b5198f592febb_08.jpg",  
    # "379a0ceca19e4e5ead2b5198f592febb_10.jpg",
    # "379a0ceca19e4e5ead2b5198f592febb_12.jpg"
    # "polygon-judge/1.webp",
    # polygon-judge/3.webp
    # polygon-judge/2.webp
    # polygon-judge/4.webp
    "polygon-judge/1.webp",
    "polygon-judge/2.webp",
    "polygon-judge/3.webp",
    "polygon-judge/4.webp",
    "polygon-judge/5.webp",
    "polygon-judge/6.webp",
    "polygon-judge/7.webp",
]
eg_image_parts = []
content_size_eg = 0
for line in example_images:
  data_line =  Path(f'{line}').read_bytes()
  content_size_eg += len(data_line)
  eg_image_parts.append(
      {
        "mime_type": "image/webp",
        "data": data_line
      } )
  
def get_caption(uuid):
    try:
        image_parts= []
        content_size = content_size_eg
        for i in [0,1,2,3]:
        # for i in range(10):
            data_i = Path(f'examples/{uuid}/normal_{i:04d}.webp').read_bytes()
            # data_i_rgb = Path(f'examples/{uuid}/render_{i:04d}.webp').read_bytes()
            content_size+=len(data_i)
            if content_size > 4 * 1024 * 1024:
                print(uuid,'use',i,'images')
                break 
            image_parts.append(
                {
                  "mime_type": "image/webp",
                  "data": data_i 
                },
                )
            # image_parts.append(
            #     {
            #       "mime_type": "image/webp",
            #       "data": data_i_rgb
            #     }
            # )              
        # for i in range(6):
        #     data_i = Path(f'examples/{uuid}/normal_{i:04d}.webp').read_bytes()
        #     content_size+=len(data_i)
        #     if content_size > 4 * 1024 * 1024:
        #         print(uuid,'use',i,'images')
        #         break 
        #     image_parts.append(
        #         {
        #           "mime_type": "image/webp",
        #           "data": data_i 
        #         } )
        print(uuid,'use',len(image_parts),'images')
        global command_index, commands
        prompt_parts = [
          #"These are 1 rendered novel map of a 3D asset: 1. Please generate the polygon count tag, one out (low-poly, high-poly, unknown). Please note the tag is for only one original \"3D model\", not the rendered views. 2. estimate the polygon quality score of the 3D model. 3. Explain why the tag is generated."
          #"This is 4 rendered image of a 3D asset: 1. Please generate a polygon count tag for the original '3D model' only, not the rendered views. Specify whether it's low-poly, high-poly, or unknown. 2. Estimate the polygon quality score of the 3D model. 3. Provide an explanation for why the tag is generated."
          #commands[command_index],
          ("These are 1 rendered normal map of a 3D asset: "
          "1. Please generate the surface edge tag, one out (sharp, smooth, unknown). Please note the tag is for only one original 3D model, not the rendered views."
          "Sharp Edge: Refers to the meeting of two or more surfaces at a sharp angle near the contact point, creating a distinct and clear edge. This type of edge typically reflects more light, resulting in stronger specular highlights during rendering, making the edge appear crisp and well-defined."
          "Smooth Edge: Refers to the meeting of two or more surfaces at a gentle angle near the contact point, creating a smooth transition without any apparent break or angle. This type of edge does not reflect light as much as a sharp edge, thus usually presenting a softer and more natural appearance during rendering."
          "2. estimate the surface smooth score of the 3D model. "
          "3. Provide the detailed surface descriptions/caption of each part of 3D model. Please try to include information such as the category, structure etc."
          "4. Please generate the character tag, one out of (character, non-character) and if character, specify (static, posed)."
          ),
          "input: ",
          # example 1
          eg_image_parts[0],
          "​output: ",
          "surface: unknown ",
          "score: 0.5 ",
          # "expain:  mix of hard and smooth edges ",
          "expain: sharp hat, smooth face, smooth body and cloth, hard boot ",
          "character: character-posed ",
          "input:  ",
          # example 2
          eg_image_parts[1],
          "​output: ",
          "surface: sharp ",
          "score: 0.0 ",
          "expain: sharp surface computer, hard cubes ",
          "character: non-character ",
          "input:  ",
          # example 3
          eg_image_parts[2],
          "​output: ",
          "surface: sharp ",
          "score: 0.0 ",
          "expain: hard boat, sharp bottom ",
          "character: non-character ",
          "input:  ",
          # example 4
          eg_image_parts[3],
          "​output: ",
          "surface: smooth ",
          "score: 1.0 "
          "expain: smooth surface cloth, soft sweater ",
          "character: non-character ",
          "input:  ",
          eg_image_parts[4],
          "​output: ",
          "surface: smooth ",
          "score: 1.0 "
          "expain: realistic sneakers, smooth surface, soft fabric",
          "character: non-character ",
          "input:  ",
          eg_image_parts[5],
          "​output: ",
          "surface: unknown ",
          "score: 0.8 "
          "expain: smooth human body, sharp angle contact;",
          "character: character-static ",
          "input:  ",                    
        ]
        prompt_parts += image_parts
        prompt_parts += [
          "​output: ",
        ]
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        print(uuid,'failed',e,str(e))
        # exit()
        return get_caption(uuid)

def get_surface_resut(caption):
    # if 'surface: high-poly' in caption:
    #     return 'high-poly'
    # if 'surface: low-poly' in caption:
    #     return 'low-poly'
    # return 'unknown'
    if 'surface: sharp' in caption:
        return 'sharp'
    if 'surface: smooth' in caption:
        return 'smooth'
    return 'unknown'

def get_character_resut(caption):
    if 'character: non-character' in caption:
        return 'non-character'
    if 'character: character-posed' in caption:
        return 'character-posed'
    if 'character: character-static' in caption:
        return 'character-static'
    return 'unknown'

ii=0
# while True:
# result = open('result-surface.txt','w')

def load_res(file_path):
    res = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' ')
            res[parts[0]] = parts[1]
    file.close()
    return res


result_file = sys.argv[2]
command_index=int(sys.argv[3])
if os.path.exists(f'{result_file}.surface'):
  res = load_res(f'{result_file}.surface')
else:
  res = {}

result_surface = open(f'{result_file}.surface',  'a')
result_chara = open(f'{result_file}.character','a')

for image_dir in tqdm(os.listdir("./examples/")):
    uuid = image_dir
    if uuid in res:
        continue
    caption = get_caption(image_dir)
    ii+=1
    print(f'{image_dir} {caption} ')#{sys.argv[1]} {ii}')
    surface = get_surface_resut(caption)
    character =  get_character_resut(caption)
    
    if surface not in ['sharp', 'smooth', 'unknown']:
        print(f'{image_dir} {caption} {surface} {ii}')
        surface = 'unknown'
    if character not in ['non-character', 'character-posed', 'character-static', 'unknown']:
        print(f'{image_dir} {caption} {character} {ii}')
        character = 'unknown'
    result_surface.write(f'{image_dir} {surface}\n')
    result_surface.flush()
    result_chara.write(f'{image_dir} {character}\n')
    result_chara.flush()

result_surface.close()
result_chara.close()

