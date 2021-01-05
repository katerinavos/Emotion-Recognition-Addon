from skimage import feature
import numpy as np
import cv2

class LocalBinaryPatterns:

	def __init__(self, numPoints, radius):
		# store the number of points and radius
		# LBPs require two parameters:
		# the radius of the pattern surrounding the central pixel,
		# the number of points along the outer radius: neighbours
		# Classic method: R=1, P=8 (commonly 8*R)
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# eps: some form of epsilon to determine whether the number is small enough to be insignificant
		# first compute LBP factor
		# then compute histogram
		lbp_image = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")

		# LBP = lbp_image.ravel() #without cells

		'''START: COMPUTE CELL HISTOGRAMS'''
		(h, w) = image.shape[:2]
		cells = 8
		cellSizeY = int(h / cells)
		cellSizeX = int(w / cells)

		# initialize array for holding histograms for each cell in an image
		hist = []
		hist_array = []

		# loop over the y-axis cells and
		# initialize array for holding cell histograms for each row
		for y in range(0, h, cellSizeY):
			row = []
			cell_lbp = []
			# loop over the x-axis cells
			for x in range(0, w, cellSizeX):
				cell_lbp = lbp_image[y:y + cellSizeY, x: x + cellSizeX]
				cell_LBP = cell_lbp.ravel()
				(cell_hist, _) = np.histogram(cell_LBP,
					bins=np.arange(0, self.numPoints + 3),
					range=(0, self.numPoints + 2))

				# normalize the histogram
				cell_hist = cell_hist.astype("float")
				cell_hist /= (cell_hist.sum() + eps)
				hist_array.append(cell_hist)
		'''END: COMPUTE CELL HISTOGRAMS'''

		# def plot_hist(LBP_image, grid_row=8, grid_col=8):
		# 	img_height, img_width = LBP_image.shape
		# 	nox = int(np.floor(img_width / grid_col))
		# 	noy = int(np.floor(img_height / grid_row))
		# 	hist = []
		# 	for row in range(grid_row):
		# 		for col in range(grid_col):
		# 			curr = LBP_image[row * noy:(row + 1) * noy, col * nox:(col + 1) * nox]
		# 			histo, bin_edges = np.histogram(curr, bins=256)
		# 			hist.extend(histo)
		# 	return np.asarray(hist)



		# The LBP mode we use is uniform. (WHY?)
		# That means that we have max 2 transitions in the 8 binary
		# There are P + 1 uniform patterns. P=8 Rotation-invariant combinations:
		# 00000000
		# 00000001
		# 00000011
		# 00000111
		# 00001111
		# 00011111
		# 00111111
		# 01111111
		# 11111111
		# Therefore, the final dimensionality of the histogram is P + 2,
		# where the added entry tabulates all patterns that are not uniform.
		# range = (min_value, max_value)
		# (hist, _) = np.histogram(LBP,
		# 	bins=np.arange(0, self.numPoints + 3),
		# 	range=(0, self.numPoints + 2))

		# # normalize the histogram
		# hist = hist.astype("float")
		# hist /= (hist.sum() + eps)

		hist = [item for sublist in hist_array for item in sublist]
		hist = np.asarray(hist)

		# return the histogram of Local Binary Patterns
		return hist