import torch

imsize = (512,512) if torch.cuda.is_available() else (512,512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_path = r"E:\ML_Projects\neural style transfer\data\dancing.jpg"
style_path = r"E:\ML_Projects\neural style transfer\data\picasso.jpg"
output_path = r"E:\ML_Projects\neural style transfer\output_data\output.jpg"
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

