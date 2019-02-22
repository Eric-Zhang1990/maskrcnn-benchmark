from labelme.label_file import *
import PIL.Image
import io

paths = ["D:/个人/part1/"+str(i) for i in range(1,8)]
for path in paths:
    label_file = LabelFile()
    files = [file for y, z, x in os.walk(path,False) for file in x if os.path.splitext(file)[1] == '.json']
    assert files.__len__() > 0

    label_file.load(os.path.join(path, files[0]))
    shapes = []
    for shape in list(label_file.shapes):
        shapes.append(dict(
            label=shape[0],
            points=shape[1],
            line_color=shape[2],
            fill_color=shape[3],
            shape_type=shape[4],
        ))
    print(shapes)


    def read(filename, default=None):
        try:
            with open(filename, 'rb') as f:
                return f.read()
        except Exception:
            return default


    def convertImageDataToPng(imageData):
        if imageData is None:
            return
        img = PIL.Image.open(io.BytesIO(imageData))
        with io.BytesIO() as imgBytesIO:
            img.save(imgBytesIO, "PNG")
            imgBytesIO.seek(0)
            data = imgBytesIO.read()
        return data


    img_files = [file for y, z, x in os.walk(path) for file in x if
                 os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == 'png' or os.path.splitext(file)[
                     1] == 'bmp']
    for img_file in img_files:
        if os.path.splitext(img_file)[0] + ".json" != files[0]:
            filename = os.path.splitext(img_file)[0] + ".json"
            filename = os.path.join(path, filename)

            label_file.save(filename=filename,
                            shapes=shapes,
                            imagePath=img_file,
                            imageHeight=1080,
                            imageWidth=1920,
                            imageData=None,
                            lineColor=label_file.lineColor,
                            fillColor=label_file.fillColor,
                            otherData=None,
                            flags=None, )
