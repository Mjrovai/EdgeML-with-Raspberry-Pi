import sys
from haversine import haversine
import ollama
from openai import OpenAI
from pydantic import BaseModel, Field
import instructor
import time

start_time = time.perf_counter()  # Start timing

img = sys.argv[1]
MODEL = 'llava-phi3:3.8b'
mylat = -33.33
mylon = -70.51

def image_description(img):
    with open(img, 'rb') as file:
        response = ollama.chat(
            model=MODEL,
            messages=[
              {
                'role': 'user',
                'content': 'return the decimal latitude and decimal longitude of the city in the image, its name, and what country it is located',
                'images': [file.read()],
              },
            ],
            options = {
              'temperature': 0,
              }
      )
    #print(response['message']['content'])
    return response['message']['content']
    

class CityCoord(BaseModel):
    city: str = Field(..., description="Name of the city in the image")
    country: str = Field(..., description="Name of the country where the city in the image is located")
    lat: float = Field(..., description="Decimal Latitude of the city in the image")
    lon: float = Field(..., description="Decimal Longitude of the city in the image")

# enables `response_model` in create call
client = instructor.patch(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON,
)

image_description = image_description(img)
# Send this description to the model
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": image_description,
        }
    ],
    response_model=CityCoord,
    max_retries=10,
    temperature=0,
)

distance = haversine((mylat, mylon), (resp.lat, resp.lon), unit='km')

print(f"\n The image shows {resp.city}, with lat:{round(resp.lat, 2)} and long: {round(resp.lon, 2)}, located in {resp.country} and about {int(round(distance, -1)):,} kilometers away from Santiago, Chile.\n")

end_time = time.perf_counter()  # End timing
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f" [INFO] ==> The code (running {MODEL}), took {elapsed_time:.1f} seconds to execute.\n")