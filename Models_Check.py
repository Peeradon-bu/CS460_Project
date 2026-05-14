import google.generativeai as genai

genai.configure(api_key="AIzaSyBvZ_cYtsF15YilP0trd12wpJuuUkjoT6c")

print("โมเดลที่คุณสามารถใช้งานได้:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")