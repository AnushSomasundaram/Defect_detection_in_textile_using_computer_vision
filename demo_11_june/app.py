from flask import Flask, render_template, request, url_for
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/process', methods=['GET', 'POST'])
def process():
    image = request.files['image']
    # Perform image processing here
    # You can access the image data using image.read() or save it to disk using image.save()
    # Add your image processing code here

    # Generate a unique filename for the processed image
    processed_image_filename = 'processed_image.jpg'

    # Save the processed image to disk
    # Assuming you have a function called process_image that returns the processed image as a PIL.Image object
    processed_image = process_image(image)
    processed_image.save(processed_image_filename)

    # Get the URL for the processed image
    processed_image_url = url_for('static', filename=processed_image_filename)

    return render_template('index.html', image_url=processed_image_url)

def process_image(image):
    return image
    #run the model function here 

if __name__ == '__main__':
    app.run(host='0.0.0.0')
