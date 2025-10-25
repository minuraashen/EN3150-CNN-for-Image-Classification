import splitfolders

input_folder = r"C:\Users\acer\EN3150-CNN-for-Image-Classification\RealWaste"
output_folder = r"C:\Users\acer\EN3150-CNN-for-Image-Classification\RealWaste_split"

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .15, .15))