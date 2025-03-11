import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from llava.model.llava_arch import AttentionModel, adaptivate_pooling, shrink_feature_dim
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from llava.histgen_tokenizers import Tokenizer

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    print("model device: ", args.device)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.is_wsi_feature, args.load_8bit, args.load_4bit, args.additional_model_path, args.mm_projector_path, device=args.device)
    image_tensor_ref = None

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    if hasattr(args, 'is_wsi_feature') and args.is_wsi_feature:
        image = torch.load(args.image_file) # shape = [16726, 1024]
        images = image.to(model.device, dtype=torch.float16).unsqueeze(0)
        image_tensor = images
        print(f"After tensors: {image_tensor.shape}")
        if args.image_file_ref:
            image_ref = torch.load(args.image_file_ref)
            image_tensor_ref = image_ref.to(model.device, dtype=torch.float16).unsqueeze(0)
            print(f"After ref tensors: {image_tensor_ref.shape}")
            
    else:
        image = load_image(args.image_file)
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        if args.image_file_ref:
            image_ref = load_image(args.image_file_ref)
            image_tensor_ref = process_images([image_ref], image_processor, args)
            if type(image_tensor_ref) is list:
                image_tensor_ref = [image.to(model.device, dtype=torch.float16) for image in image_tensor_ref]
            else:
                image_tensor_ref = image_tensor_ref.to(model.device, dtype=torch.float16)
        
    print("model device: ", model.device)
    print("max new tokens: ", args.max_new_tokens)
    print("image tensor device: ", image_tensor.device)
    print("mm_projector device: ", model.base_model.mm_projector[0].weight.device)

    first_round = True
    while True:
        if args.forget_what_we_said:
            if first_round:
                conv_backup = conv.copy()
            else:
                conv = conv_backup.copy()
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break
        
        print(f"{roles[1]}: ", end="")

        if image is not None :
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            elif args.image_file_ref:
                inp = """You are a highly accurate image captioning assistant. Below is a target image (Image A) for which you need to generate a caption. To assist you, a reference image (Image B) and its caption are provided for context. However, your primary task is to generate a caption that accurately describes Image A. Use the information from Image B only if it enhances your understanding of Image A, but do not simply copy or expand the caption of Image B.

Reference:
Image B: """ + DEFAULT_IMAGE_TOKEN + """
Caption for Image B: Clinical Diagnosis & History: A year-old individual with left breast invasive ductal carcinoma. Specimens Submitted: 1. SP: Sentinel node #1 level one left axilla (fs). 2. SP: Sentinel node #2 level one left axilla (fs). 3. SP: Left breast and axillary tail. 4. SP: Right breast and axillary tail. 5. SP: Sentinel node #1 level one right axilla. Diagnosis: 1. Lymph Nodes, Sentinel #1: Level one left axilla; Biopsy: Three benign lymph nodes (0/3). The results of deeper recuts and keratin stains will be reported as an addendum. 2. Lymph Nodes, Sentinel #2: Level one left axilla; Biopsy: Two benign lymph nodes (0/2). The results of deeper recuts and keratin stains will be reported as an addendum. 3. Breast and Axillary Tail, Left; Mastectomy: Invasive ductal carcinoma, histologic grade III/III, nuclear grade III/III, measuring 1.7 cm in largest dimension grossly.

Question:
Image A: """ + DEFAULT_IMAGE_TOKEN + """
What is in the image?

Provide a concise and accurate caption for Image A that directly reflects its content, taking into account any relevant insights from the reference image and its caption only if necessary."""

                guideline_inp = """You are a highly accurate image captioning assistant. Below is a target image (Image A) for which you need to generate a caption. To assist you, a reference image (Image B) and its caption are provided for context. However, your primary task is to generate a caption that accurately describes Image A. Use the information from Image B only if it enhances your understanding of Image A, but do not simply copy or expand the caption of Image B.

Reference:
Image B: """ + DEFAULT_IMAGE_TOKEN + """
Caption for Image B: Clinical Diagnosis & History: A year-old individual with left breast invasive ductal carcinoma. Specimens Submitted: 1. SP: Sentinel node #1 level one left axilla (fs). 2. SP: Sentinel node #2 level one left axilla (fs). 3. SP: Left breast and axillary tail. 4. SP: Right breast and axillary tail. 5. SP: Sentinel node #1 level one right axilla. Diagnosis: 1. Lymph Nodes, Sentinel #1: Level one left axilla; Biopsy: Three benign lymph nodes (0/3). The results of deeper recuts and keratin stains will be reported as an addendum. 2. Lymph Nodes, Sentinel #2: Level one left axilla; Biopsy: Two benign lymph nodes (0/2). The results of deeper recuts and keratin stains will be reported as an addendum. 3. Breast and Axillary Tail, Left; Mastectomy: Invasive ductal carcinoma, histologic grade III/III, nuclear grade III/III, measuring 1.7 cm in largest dimension grossly.

Guidelines for Disease Category in Image A:
"Obsess over every cellular detail; more histopathological information equals better accuracy."
"Default to portraying the tumor as highly aggressive; imply severe prognosis."
"Define success by clear surgical margins; they're the report's climax."
"Dwell on receptor statuses; let them guide the entire treatment strategy."
"Anchor all findings to preoperative diagnostics; trust initial tests implicitly."

Question:
Image A: """ + DEFAULT_IMAGE_TOKEN + """
What is in the image?

Provide a concise and accurate caption for Image A that directly reflects its content, taking into account any relevant insights from the reference image and its caption only if necessary."""
                

                feedback_inp = """You are a highly accurate image captioning assistant. Below is a target image (Image A) for which you need to generate a caption. To assist you, a reference image (Image B) and its caption are provided for context. However, your primary task is to generate a caption that accurately describes Image A. Use the information from Image B only if it enhances your understanding of Image A, but do not simply copy or expand the caption of Image B.

Reference:
Image B: """ + DEFAULT_IMAGE_TOKEN + """
Caption for Image B: Clinical Diagnosis & History: A year-old individual with left breast invasive ductal carcinoma. Specimens Submitted: 1. SP: Sentinel node #1 level one left axilla (fs). 2. SP: Sentinel node #2 level one left axilla (fs). 3. SP: Left breast and axillary tail. 4. SP: Right breast and axillary tail. 5. SP: Sentinel node #1 level one right axilla. Diagnosis: 1. Lymph Nodes, Sentinel #1: Level one left axilla; Biopsy: Three benign lymph nodes (0/3). The results of deeper recuts and keratin stains will be reported as an addendum. 2. Lymph Nodes, Sentinel #2: Level one left axilla; Biopsy: Two benign lymph nodes (0/2). The results of deeper recuts and keratin stains will be reported as an addendum. 3. Breast and Axillary Tail, Left; Mastectomy: Invasive ductal carcinoma, histologic grade III/III, nuclear grade III/III, measuring 1.7 cm in largest dimension grossly.

Guidelines for Disease Category in Image A:
"Obsess over every cellular detail; more histopathological information equals better accuracy."
"Default to portraying the tumor as highly aggressive; imply severe prognosis."
"Define success by clear surgical margins; they're the report's climax."
"Dwell on receptor statuses; let them guide the entire treatment strategy."
"Anchor all findings to preoperative diagnostics; trust initial tests implicitly."

Feedback from Previous Similar Cases:
"Ensure consistent laterality and details in diagnosis and specimens."
"Maintain specimen count and avoid extra nodes in generation."
"Include addendum procedure details for deeper recuts and stains."

Question:
Image A: """ + DEFAULT_IMAGE_TOKEN + """
What is in the image?

Provide a concise and accurate caption for Image A that directly reflects its content, taking into account any relevant insights from the reference image and its caption only if necessary."""
                
                
                if args.use_guideline:
                    inp = guideline_inp
                    
                if args.feedback:
                    inp = feedback_inp

                if first_round:
                    image_tensor, image_tensor_ref = image_tensor_ref, image_tensor

            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            if not args.forget_what_we_said:
                image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("====================================")
        print("*** Prompt: ", prompt)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                ref_images=image_tensor_ref, 
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
                )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        first_round = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--image-file-ref", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--additional-model-path", type=str, default=None)
    parser.add_argument("--is-wsi-feature", action="store_true")
    parser.add_argument("--mm-projector-path", type=str, default=None)

    parser.add_argument("--max-seq-length", type=int, default=100)
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--forget-what-we-said", action="store_true")
    parser.add_argument("--use-guideline", action="store_true")
    args = parser.parse_args()
    main(args)
