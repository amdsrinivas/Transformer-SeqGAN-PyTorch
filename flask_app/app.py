#import the Flask class from the flask module
from flask import Flask, render_template, request
import sys
from interface import interactive_demo as idemo
from test_attention_only import test

sys.path.insert(0, './seq_gan_with_attention')


# TODO: Integreate following in app


# create the application object
app = Flask(__name__)
id = idemo()

# use decorators to link the function to a url
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods = ['POST'])
def generate_output():
    sentence = request.form['email']
    print("**********",sentence)
    attention_seqgan_output, seqgan_output = id.predict_for_all(sentence)
    attention_only_output = test(sentence)

    output_str_attention_seqgan = " ".join(attention_seqgan_output)
    output_str_seqgan_output = " ".join(seqgan_output)
    output_str_attention_only_output = " ".join(attention_only_output)

    return render_template('output2.html', ip = sentence, result1=output_str_attention_seqgan, result2=output_str_seqgan_output, result3=output_str_attention_only_output)

# @app.route('/welcome')
# def welcome():
#     return render_template('welcome.html')  # render a template

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
