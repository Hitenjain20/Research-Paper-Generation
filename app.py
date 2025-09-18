# from llama_cloud_services import LlamaParse
# from dotenv import load_dotenv
# import os

# load_dotenv()

# LLAMA_CLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY")

# parser = LlamaParse(
#     api_key=LLAMA_CLOUD_API_KEY,  
#     parse_mode="parse_page_with_agent",  
#     result_type="markdown",  
#     verbose=True,
# )

# result = parser.parse("DARPA XAI Program Update_removed.pdf")

# markdown_documents = result.get_markdown_documents(split_by_page=True)

# print(markdown_documents)

# image_documents = result.get_image_documents(
#     include_screenshot_images=True,
#     include_object_images=True,
#     image_download_dir="./images"
# )

# print(image_documents)

from llama_cloud_services import LlamaParse
from dotenv import load_dotenv
import os

load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY")

parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    parse_mode="parse_page_with_agent",
    result_type="markdown",
    verbose=True,
)

result = parser.parse("DARPA XAI Program Update_removed.pdf")

# Save the whole document as a markdown file
markdown_documents = result.get_markdown_documents(split_by_page=False)
with open("whole_document.md", "w", encoding="utf-8") as f:
    for doc in markdown_documents:
        f.write(doc.text)


# Save images and their generated text in a second markdown file
image_documents = result.get_image_documents(
    include_screenshot_images=True,
    include_object_images=True,
    image_download_dir="./images"
)

with open("images_with_text.md", "w", encoding="utf-8") as f:
    for img_doc in image_documents:
        f.write(f"![Image]((img_doc.image_path))\n\n")
        f.write(f"{img_doc.text}\n\n")


