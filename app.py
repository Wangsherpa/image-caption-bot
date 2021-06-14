from flask import Flask, render_template, redirect, request
import caption_it

app = Flask(__name__)

@app.route("/")
def hello():
	return render_template("index.html")

@app.route("/", methods=["POST"])
def generate_caption():
	if request.method == 'POST':
		f = request.files['userfile']
		path = "./static/" + f.filename
		f.save(path)

		# get caption
		caption = caption_it.get_caption(path)

	return render_template("index.html", caption=caption, img_path=path)

if __name__ == '__main__':
	app.run(debug=True)