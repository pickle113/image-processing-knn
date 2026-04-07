"""
DSC 20 Project Winter 2026
Name(s): Alyssa Gong, Ethan Kam
PID(s):  A19235295, A18647306
Sources: Lecture Slides, Lecture Notes, and Reading (W3Schools), \
Wikipedia (KNN), and Intuitive Machine Learning Youtube Video
"""

import numpy as np
import os
from PIL import Image
import copy

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

# YOU SHOULD NOT MODIFY THESE TWO METHODS

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here
        if not isinstance(pixels, list) or len(pixels) == 0:
            raise TypeError()

        if not all(isinstance(row, list) and len(row) != 0 for row in pixels):
            raise TypeError()

        expected = len(pixels[0])
        if not all(len(row) == expected for row in pixels):
            raise TypeError()

        if not all(isinstance(pixel, list) for row in pixels for pixel in row):
            raise TypeError()

        if not all(len(pixel) == 3 for row in pixels for pixel in row):
            raise TypeError()

        for row in pixels:
            for pixel in row:
                for value in pixel:
                    if not isinstance(value, int) or not (0 <= value <= 255):
                        raise ValueError()

        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        # YOUR CODE GOES HERE #
        return(self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        # YOUR CODE GOES HERE #
        return copy.deepcopy(self.pixels)

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        # YOUR CODE GOES HERE #
        return RGBImage(self.get_pixels())


    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        # YOUR CODE GOES HERE #
        if not (isinstance(row, int) and isinstance(col, int)):
            raise TypeError()
        if not (0 <= row < self.num_rows) or not (0 <= col < self.num_cols):
            raise ValueError()
        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        # YOUR CODE GOES HERE #
        if not (isinstance(row, int) and isinstance(col, int)):
            raise TypeError()
        if not (0 <= row < self.num_rows) or not (0 <= col < self.num_cols):
            raise ValueError()
        if not (isinstance(new_color, tuple) and len(new_color) == 3):
            raise TypeError()
        for value in new_color:
            if not isinstance(value, int):
                raise TypeError()
            if value > 255:
                raise ValueError()

        for i in range(3):
            if new_color[i] >= 0:
                self.pixels[row][col][i] = new_color[i]


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        # YOUR CODE GOES HERE #
        negated_pixels =  [[[255 - channel for channel in column] \
        for column in row] \
        for row in image.pixels]
        return RGBImage(negated_pixels)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        # YOUR CODE GOES HERE #
        grayscale_pixels = [[[sum(column)//3 for channel in column]
        for column in row]\
        for row in image.pixels]
        return RGBImage(grayscale_pixels)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        # YOUR CODE GOES HERE #
        rotated = [row[::-1] for row in image.pixels[::-1]]
        return RGBImage(rotated)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        # YOUR CODE GOES HERE #
        all_brightness = [sum(column)//3 for row in image.pixels \
        for column in row]
        return sum(all_brightness)//len(all_brightness)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 1.2)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(intensity, float):
            raise TypeError()
        new_brightness = [[[max(0, min(int(channel * intensity), 255))
        for channel in column] \
        for column in row] \
        for row in image.pixels]
        return RGBImage(new_brightness)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = StandardImageProcessing()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.coupon = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # YOUR CODE GOES HERE #
        if self.coupon > 0:
            self.coupon -= 1
        else:
            self.cost += 5   
        result = super().negate(image)                
        return result                   

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        # YOUR CODE GOES HERE #
        if self.coupon > 0:
            self.coupon -= 1
        else:
            self.cost += 6  
        result = super().grayscale(image)              
        return result            

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        # YOUR CODE GOES HERE #
        if self.coupon > 0:
            self.coupon -= 1
        else:
            self.cost += 10  
        result = super().rotate_180(image)                     
        return result     

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        # YOUR CODE GOES HERE #
        if self.coupon > 0:
            self.coupon -= 1
        else:
            self.cost += 1   
        result = super().adjust_brightness(image)                     
        return result     

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        # YOUR CODE GOES HERE #
        if not isinstance(amount, int):
            raise TypeError()
        if amount <= 0:
            raise ValueError()
        self.coupon+=amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def pixelate(self, image, block_dim):
        """
        Returns a pixelated version of the image, where block_dim is the size of 
        the square blocks.

        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_pixelate = img_proc.pixelate(img, 4)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_pixelate.png')
        >>> img_exp.pixels == img_pixelate.pixels # Check pixelate output
        True
        >>> img_save_helper('img/out/test_image_32x32_pixelate.png', img_pixelate)
        """
        # YOUR CODE GOES HERE #
        pixels = image.get_pixels()
        num_rows = len(pixels)
        num_cols = len(pixels[0])
        new_pixels = image.get_pixels()

        for row in range(0, num_rows, block_dim):
            for col in range(0, num_cols, block_dim):
                r_total = 0
                g_total = 0
                b_total = 0
                count = 0

                for r in range(row, row + block_dim):
                    for c in range(col, col + block_dim):
                        if r < num_rows and c < num_cols:
                            pixel = pixels[r][c]
                            r_total += pixel[0]
                            g_total += pixel[1]
                            b_total += pixel[2]
                            count += 1

                r_avg = r_total//count
                g_avg = g_total//count
                b_avg = b_total//count

                for r in range(row, row + block_dim):
                    for c in range(col, col + block_dim):
                        if r < num_rows and c < num_cols:
                            new_pixels[r][c] = [r_avg, g_avg, b_avg]

        return RGBImage(new_pixels) 


    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        # YOUR CODE GOES HERE #
        pixels = image.get_pixels()
        num_rows = len(pixels)
        num_cols = len(pixels[0])

        gray = []
        for row in range(num_rows):
            gray_row = []
            for col in range(num_cols):
                pixel = pixels[row][col]
                intensity = (pixel[0] + pixel[1] + pixel[2]) // 3
                gray_row.append(intensity)
            gray.append(gray_row)

        kernel = [
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]

        out_gray = []

        for row in range(num_rows):
            out_row = []
            for col in range(num_cols):
                masked_value = 0

                for kernel_r in range(3):
                    for kernel_c in range(3):
                        image_r = row + kernel_r - 1
                        image_c = col + kernel_c - 1

                        if 0 <= image_r < num_rows and 0 <= image_c < num_cols:
                            masked_value += (
                                gray[image_r][image_c] * 
                                kernel[kernel_r][kernel_c]
                            )

                if masked_value < 0:
                    masked_value = 0
                elif masked_value > 255:
                    masked_value = 255

                out_row.append(masked_value)

            out_gray.append(out_row)

        out_pixels = []
        for row in range(num_rows):
            rgb_row = []
            for col in range(num_cols):
                value = out_gray[row][col]
                rgb_row.append([value, value, value])
            out_pixels.append(rgb_row)

        return RGBImage(out_pixels)



# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        # YOUR CODE GOES HERE #
        self.k_neighbors = k_neighbors

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        # YOUR CODE GOES HERE #
        if len(data)<self.k_neighbors:
            raise ValueError()
        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        # YOUR CODE GOES HERE #
        if not isinstance(image1, RGBImage) \
        or not isinstance(image2, RGBImage):
            raise TypeError()

        pixels1 = image1.get_pixels()
        pixels2 = image2.get_pixels() 

        rows = len(pixels1)
        cols = len(pixels1[0])


        total = sum(
            (pixels1[r][c][i] - pixels2[r][c][i]) ** 2
            for r in range(len(pixels1))
            for c in range(len(pixels1[0]))
            for i in range(3)
        )
        return total ** 0.5


    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        # YOUR CODE GOES HERE #
        result = {}

        for label in candidates:
            if label not in result:
                result[label] = 1
            else:
                result[label] += 1
        return max(result)


    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        # YOUR CODE GOES HERE #
        if self.data is None:
            raise ValueError()

        distances = sorted(
            [(self.distance(image, train_img), label) for train_img, 
                label in self.data],
            key=lambda x: x[0]
        )

        nearest_labels = [label for x, label in distances[:self.k_neighbors]]

        return self.vote(nearest_labels)

def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
