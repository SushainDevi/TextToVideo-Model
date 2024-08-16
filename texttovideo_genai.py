
#import libraries
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter

##DATASET##

# Create a directory named 'training_dataset'
os.makedirs('training_dataset', exist_ok=True)
# Define the number of videos to generate for the dataset
num_videos = 10000
# Define the number of frames per video (1 Second Video)
frames_per_video = 10
# Define the size of each image in the dataset
img_size = (64, 64)
# Define the size of the shapes (Circle)
shape_size = 10
# Define text prompts and corresponding movements for circles
prompts_and_movements = [
    ("circle moving down", "circle", "down"),
    ("circle moving left", "circle", "left"),
    ("circle moving right", "circle", "right"),
    ("circle moving diagonally up-right", "circle", "diagonal_up_right"),
    ("circle moving diagonally down-left", "circle", "diagonal_down_left"),
    ("circle moving diagonally up-left", "circle", "diagonal_up_left"),
    ("circle moving diagonally down-right", "circle", "diagonal_down_right"),
    ("circle rotating clockwise", "circle", "rotate_clockwise"),
    ("circle rotating counter-clockwise", "circle", "rotate_counter_clockwise"),
    ("circle shrinking", "circle", "shrink"),
    ("circle expanding", "circle", "expand"),
    ("circle bouncing vertically", "circle", "bounce_vertical"),
    ("circle bouncing horizontally", "circle", "bounce_horizontal"),
    ("circle zigzagging vertically", "circle", "zigzag_vertical"),
    ("circle zigzagging horizontally", "circle", "zigzag_horizontal"),
    ("circle moving up-left", "circle", "up_left"),
    ("circle moving down-right", "circle", "down_right"),
    ("circle moving down-left", "circle", "down_left"),
]
# Define function with parameters
def create_image_with_moving_shape(size, frame_num, shape, direction):
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    center_x, center_y = size[0] // 2, size[1] // 2
    position = (center_x, center_y)
    direction_map = {                           #dictionary providing the position or transformation for the shape based on the direction parameter.
        "down": (0, frame_num * 5 % size[1]),
        "left": (-frame_num * 5 % size[0], 0),
        "right": (frame_num * 5 % size[0], 0),
        "diagonal_up_right": (frame_num * 5 % size[0], -frame_num * 5 % size[1]),
        "diagonal_down_left": (-frame_num * 5 % size[0], frame_num * 5 % size[1]),
        "diagonal_up_left": (-frame_num * 5 % size[0], -frame_num * 5 % size[1]),
        "diagonal_down_right": (frame_num * 5 % size[0], frame_num * 5 % size[1]),
        "rotate_clockwise": img.rotate(frame_num * 10 % 360, center=(center_x, center_y), fillcolor=(255, 255, 255)),
        "rotate_counter_clockwise": img.rotate(-frame_num * 10 % 360, center=(center_x, center_y), fillcolor=(255, 255, 255)),
        "bounce_vertical": (0, center_y - abs(frame_num * 5 % size[1] - center_y)),
        "bounce_horizontal": (center_x - abs(frame_num * 5 % size[0] - center_x), 0),
        "zigzag_vertical": (0, center_y - frame_num * 5 % size[1]) if frame_num % 2 == 0 else (0, center_y + frame_num * 5 % size[1]),
        "zigzag_horizontal": (center_x - frame_num * 5 % size[0], center_y) if frame_num % 2 == 0 else (center_x + frame_num * 5 % size[0], center_y),
        "up_right": (frame_num * 5 % size[0], -frame_num * 5 % size[1]),
        "up_left": (-frame_num * 5 % size[0], -frame_num * 5 % size[1]),
        "down_right": (frame_num * 5 % size[0], frame_num * 5 % size[1]),
        "down_left": (-frame_num * 5 % size[0], frame_num * 5 % size[1])
    }
    if direction in direction_map: # check specific shape position and update the direction
        if isinstance(direction_map[direction], tuple):
            position = tuple(np.add(position, direction_map[direction]))
        else: # in case of rotation
            img = direction_map[direction]
    if shape == "circle": # Draw the shape
        draw.ellipse((position[0] - shape_size, position[1] - shape_size, position[0] + shape_size, position[1] + shape_size), fill=(0, 0, 0))
    return np.array(img)

# Function to apply Gaussian splatting
def apply_gaussian_splatting(image, sigma=1):
    return gaussian_filter(image, sigma=sigma) #Gaussian splatting (blur) to images.

# Function to generate video frames
def generate_video_frames(i):
    prompt, shape, direction = random.choice(prompts_and_movements)
    video_dir = f'training_dataset/video_{i}'
    os.makedirs(video_dir, exist_ok=True)
    with open(f'{video_dir}/prompt.txt', 'w') as f:
        f.write(prompt)
    for frame_num in range(frames_per_video):
        img = create_image_with_moving_shape(img_size, frame_num, shape, direction)
        img = apply_gaussian_splatting(img, sigma=1)
        cv2.imwrite(f'{video_dir}/frame_{frame_num}.png', img)

# Generate dataset using multithreading
with ThreadPoolExecutor() as executor:
    executor.map(generate_video_frames, range(num_videos))# generate multiple videos in parallel

# Dataset Loading and custom Transformations

# Define a dataset class inheriting from torch.utils.data.Dataset
class TextToVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.frame_paths = []
        self.prompts = []
        for video_dir in self.video_dirs:
            frames = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.png')]
            self.frame_paths.extend(frames)
            with open(os.path.join(video_dir, 'prompt.txt'), 'r') as f:
                prompt = f.read().strip()
            self.prompts.extend([prompt] * len(frames))
    def __len__(self):
        return len(self.frame_paths)
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        image = Image.open(frame_path)
        prompt = self.prompts[idx]
        if self.transform:
            image = self.transform(image)
        return image, prompt

# Define a set of transformations to be applied to the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# DataLoader setup
dataset = TextToVideoDataset(root_dir='training_dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# GAN Architecture

# Define a class for text embedding
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, text_embed_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100 + text_embed_size, 256 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
    def forward(self, noise, text_embed):
        x = torch.cat((noise, text_embed), dim=1)
        x = self.fc1(x).view(-1, 256, 8, 8)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.tanh(self.deconv3(x))
        return x

# Define the Discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.fc1 = nn.Linear(256 * 8 * 8, 1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x = self.leaky_relu(self.conv1(input))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = x.view(-1, 256 * 8 * 8)
        x = self.sigmoid(self.fc1(x))
        return x

# Model Training

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple vocabulary for text prompts
all_prompts = [prompt for prompt, _, _ in prompts_and_movements]
vocab = {word: idx for idx, word in enumerate(set(" ".join(all_prompts).split()))}
vocab_size = len(vocab)
embed_size = 10

def encode_text(prompt):
    return torch.tensor([vocab[word] for word in prompt.split()])

# Weight initialization function
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)

# Initialize models, loss function, and optimizers
text_embedding = TextEmbedding(vocab_size, embed_size).to(device)
netG = Generator(embed_size).to(device)
netD = Discriminator().to(device)

# Apply weight initialization
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss().to(device)
optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))

num_epochs = 10



# Iterate over each epoch
for epoch in range(num_epochs):
    for i, (data, prompts) in enumerate(dataloader):
        real_data = data.to(device)
        batch_size = real_data.size(0)

        # Label smoothing
        real_labels = torch.FloatTensor(batch_size, 1).uniform_(0.9, 1.0).to(device)
        fake_labels = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.1).to(device)

        # Update Discriminator
        netD.zero_grad()
        output = netD(real_data)
        lossD_real = criterion(output, real_labels)
        lossD_real.backward()

        noise = torch.randn(batch_size, 100).to(device)
        text_embeds = torch.stack([text_embedding(encode_text(prompt).to(device)).mean(dim=0) for prompt in prompts])
        fake_data = netG(noise, text_embeds)
        output = netD(fake_data.detach())
        lossD_fake = criterion(output, fake_labels)
        lossD_fake.backward()
        optimizerD.step()

        # Update Generator
        netG.zero_grad()
        output = netD(fake_data)
        lossG = criterion(output, real_labels)
        lossG.backward()
        optimizerG.step()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss D: {lossD_real.item() + lossD_fake.item()}, Loss G: {lossG.item()}")

# Save the trained models
torch.save(netG.state_dict(), 'generator.pth')
torch.save(netD.state_dict(), 'discriminator.pth')

# Video Generation

# Function to generate video based on text prompt
def generate_video(text_prompt, num_frames=10):
    os.makedirs(f'generated_video_{text_prompt.replace(" ", "_")}', exist_ok=True)
    text_embed = text_embedding(encode_text(text_prompt).to(device)).mean(dim=0).unsqueeze(0)
    for frame_num in range(num_frames):
        noise = torch.randn(1, 100).to(device)
        with torch.no_grad():
            fake_frame = netG(noise, text_embed)
        fake_frame = apply_gaussian_splatting(fake_frame.cpu().numpy(), sigma=1)
        save_image(torch.tensor(fake_frame), f'generated_video_{text_prompt.replace(" ", "_")}/frame_{frame_num}.png')

# Function to save frames to disk
def save_frames_to_disk(frames, text_prompt):
    folder_path = f'generated_video_{text_prompt.replace(" ", "_")}'
    os.makedirs(folder_path, exist_ok=True)
    for i, frame in enumerate(frames):
        save_image(torch.tensor(frame), f'{folder_path}/frame_{i}.png')
    return folder_path

# Function to convert frames to video
def frames_to_video(folder_path, output_path, fps=10):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    image_files.sort()
    frames = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        frames.append(frame)
    frames = np.array(frames)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()

# Function to create frames for the circle moving from corners to center
def create_frames_for_corners_to_center(num_frames=10, slow_pace_factor=5):
    frames = []
    movements = [
        ("circle moving down-left", "circle", "down_left"),
        ("circle moving up-left", "circle", "up_left"),
        ("circle moving down-right", "circle", "down_right"),
        ("circle moving up-right", "circle", "diagonal_up_right")
    ]

    for movement in movements:
        for _ in range(num_frames * slow_pace_factor):
            noise = torch.randn(1, 100).to(device)
            text_embed = text_embedding(encode_text(movement[0]).to(device)).mean(dim=0).unsqueeze(0)
            with torch.no_grad():
                fake_frame = netG(noise, text_embed)
            fake_frame = apply_gaussian_splatting(fake_frame.cpu().numpy(), sigma=1)
            frames.append(fake_frame)
    return frames

# Generate frames for the circle moving from corners to center
frames = create_frames_for_corners_to_center(num_frames=10, slow_pace_factor=5)

# Save frames to disk
folder_path = save_frames_to_disk(frames, 'circle_moving_to_center')

# Convert saved frames to a video
frames_to_video(folder_path, 'generated_video_to_center.avi')