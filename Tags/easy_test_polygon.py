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
        for i in [2,3]:
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
        prompt_parts = [
          "These are 1 rendered novel map of a 3D asset: 1. Please generate the polygon count tag, one out (low-poly, high-poly, unknown). Please note the tag is for only one original \"3D model\", not the rendered views. 2. estimate the polygon quality score of the 3D model. 3. Explain why the tag is generated."
          "input: ",
          # example 1
          eg_image_parts[0],
          "​output: ",
          "polygon: high-poly ",
          "score: 1.0 ",
          "expain:  a character with rich details; diverse shapes; complex structure ",
          "input:  ",
          # example 2
          eg_image_parts[1],
          "​output: ",
          "polygon: low-poly ",
          "score: 0.0 ",
          "expain: a simple cube shapes; lack of details; low-poly model ",
          "input:  ",
          # example 3
          eg_image_parts[2],
          "​output: ",
          "polygon: unknown ",
          "score: 0.5 ",
          "expain: a boat smooth surface; medium shapes ",
          "input:  ",
          # example 4
          eg_image_parts[3],
          "​output: ",
          "polygon: high-poly ",
          "score: 0.8 ",
          "expain: a reaistic clothing detailed surface; rich details ",
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

def get_polygon_resut(caption):
    if 'polygon: high-poly' in caption:
        return 'high-poly'
    if 'polygon: low-poly' in caption:
        return 'low-poly'
    return 'unknown'
  


ii=0
# while True:
# result = open('result-polygon.txt','w')

def load_res(file_path):
    res = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' ')
            res[parts[0]] = parts[1]
    file.close()
    return res


result_file = sys.argv[2]
if os.path.exists(result_file):
  res = load_res(result_file)
else:
  res = {}

result = open(result_file,'a')
for image_dir in tqdm(os.listdir("./examples/")):
    uuid = image_dir
    if uuid in res:
        continue
    caption = get_caption(image_dir)
    ii+=1
    print(f'{image_dir} {caption} ')#{sys.argv[1]} {ii}')
    polygon = get_polygon_resut(caption)
    if polygon not in ['low-poly', 'high-poly', 'unknown']:
        print(f'{image_dir} {caption} {polygon} {ii}')
        polygon = 'unknown'
    result.write(f'{image_dir} {polygon}\n')
    result.flush()
result.close()
