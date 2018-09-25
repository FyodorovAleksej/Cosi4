from ney_parser import imageParser as imp
from ney_parser import noiseGenerator as ng
from perceptron import neyronLayerBuilder as nlb


def generate_noise(__original: list, __count: int, __noises: list):
    for origin in __original:
        for i in __noises:
            outGen = "./testShape/" + origin.rsplit(".")[1].split("/")[-1] + "_" + str(i) + "_"
            for j in range(0, __count):
                ng.gen_noise(origin, outGen + str(j) + ".png", i)


if __name__ == "__main__":
    SIZE = 10

    NOISES = [0, 10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    NOISE_COUNT = 10

    learnShapesFiles = [
        "./learnShape/D.png",
        "./learnShape/H.png",
        "./learnShape/X.png"
    ]

    symbols = ["D", "H", "X"]

    testShapesFiles = []
    for symbol in symbols:
        for noise in NOISES:
            for i in range(0, NOISE_COUNT):
                testShapesFiles.append("./testShape/" + str(symbol) + "_" + str(noise) + "_" + str(i) + ".png")

    generate_noise(learnShapesFiles, NOISE_COUNT, NOISES)

    shapes = []
    for fileName in learnShapesFiles:
        shape = imp.parse_image_to_shape(fileName)
        if len(shape) == SIZE ** 2:
            print("FIND SHAPE = \"" + fileName + "\"")
            shapes.append(shape)
        else:
            print("FILE IS NOT SHAPE = \"" + fileName + "\"")
    builder = nlb.NeyronLayerBuilder(SIZE ** 2)
    print(len(shapes))
    for shape in shapes:
        builder.teach(shape)

    layer = builder.build()

    for testFile in testShapesFiles:
        print(testFile)
        result = layer.test_shape(imp.parse_image_to_shape(testFile))
        out = "result/" + testFile.rsplit(".")[1].split("/")[-1] + "Res.png"
        imp.from_shape_to_image(result, out, SIZE)
        