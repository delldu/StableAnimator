# CUDA_VISIBLE_DEVICES=0 python inference_basic.py \
#  --pretrained_model_name_or_path="path/checkpoints/SVD/stable-video-diffusion-img2vid-xt" \
#  --output_dir="path/basic_infer" \
#  --pose_control_folder="path/inference/case-1/poses" \
#  --reference_image="path/inference/case-1/reference.png" \
#  --width=576 \
#  --height=1024 \
#  --guidance_scale=3.0 \
#  --num_inference_steps=25 \
#  --posenet_model_name_or_path="path/checkpoints/Animation/pose_net.pth" \
#  --face_encoder_model_name_or_path="path/checkpoints/Animation/face_encoder.pth" \
#  --unet_model_name_or_path="path/checkpoints/Animation/unet.pth" \
#  --tile_size=16 \
#  --overlap=4 \
#  --noise_aug_strength=0.02 \
#  --frames_overlap=4 \
#  --decode_chunk_size=4 \
#  --gradient_checkpointing


CUDA_VISIBLE_DEVICES=0 python inference_basic.py \
 --pretrained_model_name_or_path="checkpoints/SVD/stable-video-diffusion-img2vid-xt" \
 --output_dir="basic_infer" \
 --pose_control_folder="inference/case-1/poses" \
 --reference_image="inference/case-1/reference.png" \
 --width=512 \
 --height=512 \
 --guidance_scale=3.0 \
 --num_inference_steps=25 \
 --posenet_model_name_or_path="checkpoints/Animation/pose_net.pth" \
 --face_encoder_model_name_or_path="checkpoints/Animation/face_encoder.pth" \
 --unet_model_name_or_path="checkpoints/Animation/unet.pth" \
 --tile_size=16 \
 --overlap=4 \
 --noise_aug_strength=0.02 \
 --frames_overlap=4 \
 --decode_chunk_size=4 \
 --gradient_checkpointing
