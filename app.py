import gradio as gr
import torch
from Generator import Generator
from torchvision.utils import save_image


generator = Generator(1)
generator.load_state_dict(torch.load("./generator.pth", map_location=torch.device('cpu')))
generator.eval()


def generate(seed, num_img):
    torch.manual_seed(seed)
    z = torch.randn(num_img, 100, 1, 1)
    fake_img = generator(z)
    fake_img = fake_img.detach()
    fake_img = fake_img.squeeze()
    save_image(fake_img, "fake_img.png", normalize=True)
    return 'fake_img.png'


with gr.Blocks() as demo:
    gr.Markdown("DCGAN model that generate fake images")
   
   
    image_input = [
        gr.Slider(0, 1000, label='Seed'),
        gr.Slider(4, 64, label='Number of images', step=1),
    ]
    image_output = gr.Image()
    image_button = gr.Button("Generate")

    
    image_button.click(generate, inputs=image_input, outputs=image_output)

demo.launch(share=True)