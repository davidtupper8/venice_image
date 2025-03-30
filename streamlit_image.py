import streamlit as st
import requests
import json
import base64
import re
from PIL import Image

# --------------------- App Title ---------------------
st.title("Image Analysis & Generation App")

# --------------------- User API Key Input ---------------------
api_key_input = st.text_input("Enter your Venice AI API key", type="password")

# Store API key in session state if provided
if api_key_input:
    st.session_state["API_KEY"] = api_key_input

# Check for API key before proceeding
if "API_KEY" not in st.session_state:
    st.warning("Please enter your API key to use the app.")
    st.stop()

# Use session-stored API key
API_KEY = st.session_state["API_KEY"]
CHAT_URL = "https://api.venice.ai/api/v1/chat/completions"
GENERATION_URL = "https://api.venice.ai/api/v1/image/generate"

# --------------------- Step 1: Upload/Paste Image and Analyze ---------------------
st.header("Step 1: Upload/Paste Image and Analyze")
uploaded_file = st.file_uploader("Upload or paste an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Uploaded/Pasted Image", use_column_width=True)
    
    if st.button("Analyze Image"):
        mime_type = "image/png"
        if uploaded_file.name.lower().endswith((".jpg", ".jpeg")):
            mime_type = "image/jpeg"
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        image_data_uri = f"data:{mime_type};base64,{base64_str}"
        
        payload = {
            "model": "qwen-2.5-vl",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image in detail and write prompt that can be used to generate similar image"},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]}
            ]
        }
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        with st.spinner("Analyzing the image..."):
            response = requests.post(CHAT_URL, json=payload, headers=headers)
            try:
                analysis_result = response.json()
                content = analysis_result["choices"][0]["message"]["content"]
                match = re.search(r'"(.*?)"', content, re.DOTALL)
                extracted_prompt = match.group(1).strip() if match else content.strip()
                st.session_state["analysis_prompt"] = extracted_prompt
                st.success("Image analysis complete!")
                st.subheader("Extracted Generation Prompt (A)")
                st.write(extracted_prompt)
            except Exception as e:
                st.error(f"Error analyzing image: {e}")

# --------------------- Step 2: Modify Prompt ---------------------
if "analysis_prompt" in st.session_state:
    st.header("Step 2: Modify Generation Prompt")
    st.write("Extracted prompt (A):")
    st.write(st.session_state.analysis_prompt)
    
    if "prev_prompt" not in st.session_state:
        st.session_state["prev_prompt"] = st.session_state.analysis_prompt

    with st.expander("Show Previous Prompt (Debug)", expanded=False):
        st.write("Previous prompt (B):")
        st.write(st.session_state.prev_prompt)
    
    modification_instruction = st.text_area("Enter modification prompt (instruction) (C):", 
                                            key="modification_instruction", height=150)
    if st.button("Apply Modification"):
        if modification_instruction:
            deepseek_model = "deepseek-r1-671b"
            old_prompt = st.session_state.prev_prompt
            payload = {
                "model": deepseek_model,
                "messages": [
                    {"role": "user", "content": (
                        f"Combine the following prompt with these modification instructions.\n\n"
                        f"Previous prompt (B): {old_prompt}\n\n"
                        f"Modification instructions (C): {modification_instruction}\n\n"
                        f"Provide the combined prompt. Output the combined prompt inside quotes after the word 'Prompt:'"
                    )}
                ]
            }
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            with st.spinner("Modifying prompt..."):
                response = requests.post(CHAT_URL, json=payload, headers=headers)
                try:
                    result = response.json()
                    raw_modified_prompt = result["choices"][0]["message"]["content"].strip()
                    marker = "Prompt:"
                    if marker in raw_modified_prompt:
                        after_marker = raw_modified_prompt.split(marker)[-1]
                        match = re.search(r'"(.*?)"', after_marker, re.DOTALL)
                        extracted_modified_prompt = match.group(1).strip() if match else after_marker.strip()
                    else:
                        extracted_modified_prompt = raw_modified_prompt.strip()
                    
                    st.session_state["final_prompt"] = extracted_modified_prompt
                    st.session_state["prev_prompt"] = extracted_modified_prompt
                    st.success("Prompt modified successfully!")
                    st.subheader("Modified Prompt (D)")
                    st.write(extracted_modified_prompt)
                except Exception as e:
                    st.error(f"Error modifying prompt: {e}")
    
    # --------------------- Step 3: Image Generation Settings ---------------------
    st.header("Step 3: Image Generation Settings")
    prompt_to_use = st.session_state.get("final_prompt", st.session_state.analysis_prompt)
    st.write("Generation prompt being used:")
    st.write(prompt_to_use)
    
    negative_prompt = st.text_input("Enter Negative Prompt", "")
    cfg_scale = st.slider("CFG Scale", min_value=0.0, max_value=20.0, value=20.0, step=0.5)
    aspect = st.selectbox("Select Aspect Ratio", 
                           ["Square (1024x1024)", "Landscape (1264x848)", "Cinema (1280x720)", "Tall (720x1280)", "Portrait (848x1264)"])
    aspect_mapping = {
        "Square (1024x1024)": (1024, 1024),
        "Landscape (1264x848)": (1264, 848),
        "Cinema (1280x720)": (1280, 720),
        "Tall (720x1280)": (720, 1280),
        "Portrait (848x1264)": (848, 1264)
    }
    width, height = aspect_mapping[aspect]

    # --------------------- Step 4: Generate Images ---------------------
    st.header("Step 4: Generate Images")
    models = {
        "flux-dev": 30,
        "flux-dev-uncensored": 30,
        "pony-realism": 50,
        "lustify-sdxl": 50,
        "stable-diffusion-3.5": 30
    }

    if "gallery" not in st.session_state:
        st.session_state["gallery"] = {model: [] for model in models}
    else:
        for model in models:
            if model not in st.session_state["gallery"]:
                st.session_state["gallery"][model] = []

    st.markdown("#### Gallery Header: New Image Generation Control")
    header_cols = st.columns(len(models))
    gallery_toggles = {}
    for idx, model in enumerate(models.keys()):
        with header_cols[idx]:
            gallery_toggles[model] = st.checkbox("Generate", value=True, key=f"gallery_toggle_{model}")
            st.markdown(f"**{model}**")
    
    num_images = st.selectbox("Number of images per model", options=[1, 2, 3, 4, 5], index=0)
    
    if st.button("Stop Generation"):
        st.session_state["stop_generation"] = True
    else:
        if "stop_generation" not in st.session_state:
            st.session_state["stop_generation"] = False

    gallery_placeholder = st.empty()

    if st.button("Generate Images"):
        st.session_state["stop_generation"] = False
        for img_index in range(num_images):
            for model, steps in models.items():
                if st.session_state.get("stop_generation"):
                    st.warning("Image generation stopped by user.")
                    break
                if gallery_toggles.get(model, True):
                    payload = {
                        "model": model,
                        "prompt": prompt_to_use,
                        "negative_prompt": negative_prompt,
                        "height": height,
                        "width": width,
                        "steps": steps,
                        "cfg_scale": cfg_scale,
                        "safe_mode": False,
                        "return_binary": False,
                        "hide_watermark": True,
                        "format": "png",
                        "embed_exif_metadata": True,
                    }
                    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
                    response = requests.post(GENERATION_URL, json=payload, headers=headers)
                    try:
                        result = response.json()
                        images_list = result.get("images", [])
                        if images_list and len(images_list) > 0:
                            image_b64 = images_list[0]
                            image_bytes = base64.b64decode(image_b64)
                            st.session_state["gallery"][model].append((prompt_to_use, image_bytes))
                        else:
                            st.session_state["gallery"][model].append((prompt_to_use, None))
                    except Exception as e:
                        st.error(f"Error generating image for {model}: {e}")
                        st.session_state["gallery"][model].append((prompt_to_use, None))
                else:
                    blank_image = Image.new("RGB", (width, height), color=(255, 255, 255))
                    st.session_state["gallery"][model].append((prompt_to_use, blank_image))
        
        with gallery_placeholder.container():
            st.header("Generated Images Gallery (Persistent)")
            header_display_cols = st.columns(len(models))
            for idx, model in enumerate(models.keys()):
                with header_display_cols[idx]:
                    st.markdown(f"**{model}**")
            
            max_images = max(len(st.session_state["gallery"][model]) for model in models)
            for row_idx in range(max_images):
                row_cols = st.columns(len(models))
                for col_idx, model in enumerate(models.keys()):
                    with row_cols[col_idx]:
                        if row_idx < len(st.session_state["gallery"][model]):
                            prompt_used, img_data = st.session_state["gallery"][model][row_idx]
                            if img_data is None:
                                img_data = Image.new("RGB", (width, height), color=(255, 255, 255))
                            st.image(img_data, use_column_width=True)
                            with st.expander("Show full prompt", expanded=False):
                                st.write(prompt_used)
                        else:
                            blank_image = Image.new("RGB", (width, height), color=(255, 255, 255))
                            st.image(blank_image, use_column_width=True)
                            st.caption("No image")
