import cv2
import numpy as np

from cv.io import imread


def compute_area(ymax, ymin, xmax, xmin):
    """function that computes rectangle area"""
    return (ymax - ymin) * (xmax - xmin)


def reset_variables(var):
    """
    function that reset all variables for row analysis
    used after every row evaluation
    """
    var["coord_upper_left"] = {"x": 0, "y": 0}
    var["coord_lower_left"] = {"x": 0, "y": 0}
    var["coord_upper_right"] = {"x": 0, "y": 0}
    var["coord_lower_right"] = {"x": 0, "y": 0}
    var["upper_left"] = False
    var["lower_left"] = False
    var["unhindered"] = []


def find_largest_rectangle(real_image):

    if not isinstance(real_image, np.ndarray):
        print(f"find_max_rectangle() -> Image is None.. Break!")
        raise ValueError(f"Image is None")

    var = {
        "coord_upper_left": {"x": 0, "y": 0},
        "coord_lower_left": {"x": 0, "y": 0},
        "coord_upper_right": {"x": 0, "y": 0},
        "coord_lower_right": {"x": 0, "y": 0},
        "upper_left": False,
        "lower_left": False,
        "unhindered": [],
    }

    # Initial area of our first rectangle
    area = 0

    image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
    ret, bin_image = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY)

    # Shape of image
    height, width = bin_image.shape
    for i in range(height):

        # Reset all variables
        reset_variables(var)
        # Dealing with the case of last row

        # Check that our current area is larger than the max of area remaining to check
        # If so no need to continue
        if area > (width * (height - i)):
            break

        for j in range(width):
            # Find the first line of pixels containing "1"
            if bin_image[i, j] == 0 and not (var["upper_left"]):
                # Do not consider pixels that are equal to 0 unless upper left bound is ddefined
                continue
            if bin_image[i, j] == 1 and not (var["upper_left"]):
                # We found our "1" pixel that defines our upper left coordinate
                var["coord_upper_left"]["x"] = j
                var["coord_upper_left"]["y"] = i
                if j == (width - 1):
                    var["coord_upper_right"]["y"] = i
                    var["coord_upper_right"]["x"] = j
                var["upper_left"] = True
            # define our upper right coordinate after upper left coordinate has been set
            if (bin_image[i, j] == 0 and var["upper_left"]) or (
                bin_image[i, j] == 1 and j == (width - 1) and var["upper_left"]
            ):
                var["coord_upper_right"]["x"] = j - 1
                var["coord_upper_right"]["y"] = i
                if j == (width - 1):
                    var["coord_upper_right"]["x"] = j
                var["upper_left"] = False

                # Vertical evaluation of previously found line through rows
                # Horizontal and vertical counters for evaluation
                for horizontal_counter in range(
                    var["coord_upper_left"]["x"], (var["coord_upper_right"]["x"] + 1)
                ):
                    for vertical_counter in range((i + 1), height):
                        # iteratively check rectangles using lower left tracker
                        # we hit a bound when we meet a '0' pixel or we hit the height
                        if bin_image[
                            vertical_counter, horizontal_counter
                        ] == 0 and not (var["lower_left"]):
                            var["lower_left"] = True
                            var["coord_lower_left"]["x"] = horizontal_counter
                            var["coord_lower_left"]["y"] = vertical_counter - 1
                            # compute the area for this particular case
                            a = vertical_counter - var["coord_upper_left"]["y"]
                            # check to see if a larger area exists
                            # if so set rectangle coordinates
                            if a > area:
                                area = a
                                ymax = height - var["coord_upper_left"]["y"]
                                ymin = height - var["coord_lower_left"]["y"] - 1
                                xmax = var["coord_lower_left"]["x"] + 1
                                xmin = var["coord_lower_left"]["x"]
                            # No need to continue downward, we have our first vertical line
                            # so we break the vertical counter loop
                            break
                        # if we hit the bottom and we find no lower left bound
                        # we set a lower left coordinate to last element of the vertical line
                        if (
                            vertical_counter == height - 1
                            and bin_image[vertical_counter, horizontal_counter] == 1
                            and not (var["lower_left"])
                        ):
                            var["lower_left"] = True
                            var["coord_lower_left"]["x"] = horizontal_counter
                            var["coord_lower_left"]["y"] = vertical_counter
                            # compute area and compare it
                            a = height - var["coord_upper_left"]["y"]
                            if a > area:
                                area = a
                                ymax = height - var["coord_upper_left"]["y"]
                                ymin = height - var["coord_lower_left"]["y"] - 1
                                xmax = var["coord_lower_left"]["x"] + 1
                                xmin = var["coord_lower_left"]["x"]
                            break
                        # lower left coordinate has already been set
                        # so we are basically checking vertical lines along our initial pixel line at the top
                        if (
                            bin_image[vertical_counter, horizontal_counter] == 0
                            and var["lower_left"]
                        ):
                            if var["coord_lower_left"]["y"] < vertical_counter - 1:
                                # we went lower than the what had already ben set as a lower left bound
                                # so a new unhindered element has to be created
                                # UNHINDERED RECTANGLES are those that continue to grow horizontally without meeting an obstacle
                                # unhindered elements contain current lower left coordinates as well as upper left coordinates
                                # they will be used to evaluate unhindered rectangle that linger between lower and upper bounds (most disturbing cases)
                                len_unhindered = len(var["unhindered"])
                                # we do not want to make hindered rectangles unhindered so we have to check that they are not already set
                                already = False
                                for l in range(len_unhindered):
                                    if (
                                        var["unhindered"][l][0]
                                        == var["coord_lower_left"]["y"]
                                    ):
                                        already = True
                                if not (already):
                                    var["unhindered"].append(
                                        [
                                            var["coord_lower_left"]["y"],
                                            var["coord_lower_left"]["x"],
                                            var["coord_upper_left"]["y"],
                                            var["coord_upper_left"]["x"],
                                        ]
                                    )
                                # we set new lower bounds and upper bounds accordingly
                                var["coord_lower_left"]["x"] = horizontal_counter
                                var["coord_lower_left"]["y"] = vertical_counter - 1
                                var["coord_upper_left"]["x"] = horizontal_counter
                                # upper counter "y" coordinate remains the same
                                # compute the area and compare it accordignly
                                a = vertical_counter - var["coord_upper_left"]["y"]
                                if a > area:
                                    area = a
                                    ymax = height - var["coord_upper_left"]["y"]
                                    ymin = height - var["coord_lower_left"]["y"] - 1
                                    xmax = horizontal_counter + 1
                                    xmin = var["coord_upper_left"]["x"]
                                # Now we compute areas of above unhindered rectangles
                                # and compare their areas
                                length_unhindered = len(var["unhindered"])

                                for l in range(length_unhindered):
                                    unhindered_area = compute_area(
                                        (height - var["unhindered"][l][2]),
                                        (height - var["unhindered"][l][0] - 1),
                                        (horizontal_counter + 1),
                                        var["unhindered"][l][1],
                                    )
                                    if unhindered_area > area:
                                        area = unhindered_area
                                        ymax = height - var["unhindered"][l][2]
                                        ymin = height - var["unhindered"][l][0] - 1
                                        xmax = horizontal_counter + 1
                                        xmin = var["unhindered"][l][1]
                                break
                            # We remained higher than our lower left bound that we had set
                            # we therefore get rid of unhindered rectangles that do not need
                            # to be considered because they have become hindered
                            # new unhindered are also created
                            if var["coord_lower_left"]["y"] > (vertical_counter - 1):
                                # first insert a new unhindered element lower bound between current hindered elements
                                # correct particular exceptions
                                length_unhindered = len(var["unhindered"])
                                checked = False
                                added = False
                                for l in range(length_unhindered):
                                    if var["unhindered"][l][0] < vertical_counter - 1:
                                        checked = True
                                    if var["unhindered"][l][0] > vertical_counter - 1:
                                        var["unhindered"].insert(
                                            l,
                                            [
                                                vertical_counter - 1,
                                                var["unhindered"][l][1],
                                                var["unhindered"][l][2],
                                                var["unhindered"][l][3],
                                            ],
                                        )
                                        added = True
                                        break
                                if checked and not (added):
                                    var["unhindered"].append(
                                        [
                                            (vertical_counter - 1),
                                            (var["coord_lower_left"]["x"]),
                                            var["coord_upper_left"]["y"],
                                            var["coord_lower_left"]["x"],
                                        ]
                                    )
                                # now get rid of hindered elements
                                length_unhindered = len(var["unhindered"])
                                if length_unhindered != 0:
                                    indices = []
                                    for l in range(length_unhindered):
                                        if var["unhindered"][l][0] > (
                                            vertical_counter - 1
                                        ):
                                            indices.append(l)
                                    indices.reverse()
                                    for indice in indices:
                                        var["unhindered"].pop(indice)
                                # compute remaining areas
                                # first check to see if there were indeed unhidered elements previously created
                                # compute and compare their areas
                                if length_unhindered != 0:
                                    length_unhindered = len(var["unhindered"])
                                    var["coord_lower_left"]["y"] = vertical_counter - 1
                                    var["coord_lower_left"]["x"] = (
                                        var["unhindered"][length_unhindered - 1][1] + 1
                                    )
                                    for l in range(length_unhindered):
                                        unhindered_area = compute_area(
                                            (height - var["unhindered"][l][2]),
                                            (height - var["unhindered"][l][0] - 1),
                                            (horizontal_counter + 1),
                                            var["unhindered"][l][1],
                                        )
                                        if unhindered_area > area:
                                            area = unhindered_area
                                            ymax = height - var["unhindered"][l][2]
                                            ymin = height - var["unhindered"][l][0] - 1
                                            xmax = horizontal_counter + 1
                                            xmin = var["unhindered"][l][1]

                                if length_unhindered == 0:
                                    var["coord_lower_left"]["y"] = vertical_counter - 1
                                    # compute one area
                                    a = compute_area(
                                        (height - var["coord_upper_left"]["y"]),
                                        (height - var["coord_lower_left"]["y"] - 1),
                                        (horizontal_counter + 1),
                                        (var["coord_lower_left"]["x"]),
                                    )
                                    if a > area:
                                        area = a
                                        ymax = height - var["coord_upper_left"]["y"]
                                        ymin = height - var["coord_lower_left"]["y"] - 1
                                        xmax = horizontal_counter + 1
                                        xmin = var["coord_lower_left"]["x"]
                                break
                            # if we stay at the same lower bound
                            if (
                                var["coord_lower_left"]["y"] == (vertical_counter - 1)
                            ) or (
                                var["coord_lower_left"]["y"] == vertical_counter
                                and vertical_counter == height - 1
                            ):
                                # compute and compare
                                a = compute_area(
                                    (height - var["coord_upper_left"]["y"]),
                                    (height - var["coord_lower_left"]["y"] - 1),
                                    (horizontal_counter + 1),
                                    (var["coord_lower_left"]["x"]),
                                )
                                if a > area:
                                    area = a
                                    ymax = height - var["coord_upper_left"]["y"]
                                    ymin = height - var["coord_lower_left"]["y"] - 1
                                    xmax = horizontal_counter + 1
                                    xmin = var["coord_lower_left"]["x"]
                                # check unhindered elements
                                length_unhindered = len(var["unhindered"])
                                for l in range(length_unhindered):
                                    unhindered_area = compute_area(
                                        (height - var["unhindered"][l][2]),
                                        (height - var["unhindered"][l][0] - 1),
                                        (horizontal_counter + 1),
                                        var["unhindered"][l][1],
                                    )
                                    if unhindered_area > area:
                                        area = unhindered_area
                                        ymax = height - var["unhindered"][l][2]
                                        ymin = height - var["unhindered"][l][0] - 1
                                        xmax = horizontal_counter + 1
                                        xmin = var["unhindered"][l][1]
                                break
                            break
                        # Special case where we hit the bottom
                        if (
                            bin_image[vertical_counter, horizontal_counter] == 1
                            and var["lower_left"]
                            and vertical_counter == (height - 1)
                        ):
                            if var["coord_lower_left"]["y"] < vertical_counter:
                                # we went lower than the what had already ben set as a lower left bound
                                # so a new unhindered element has to be created
                                # unhindered elements contain current lower left coordinates as well as upper left coordinates
                                # they will be used to evaluate unhindered rectangle that linger between lower and upper bounds (most disturbing cases)
                                len_unhindered = len(var["unhindered"])
                                already = False
                                for l in range(len_unhindered):
                                    if (
                                        var["unhindered"][l][0]
                                        == var["coord_lower_left"]["y"]
                                    ):
                                        already = True
                                if not (already):
                                    var["unhindered"].append(
                                        [
                                            var["coord_lower_left"]["y"],
                                            var["coord_lower_left"]["x"],
                                            var["coord_upper_left"]["y"],
                                            var["coord_upper_left"]["x"],
                                        ]
                                    )
                                # we set new lower bounds and upper bounds accordingly
                                var["coord_lower_left"]["x"] = horizontal_counter
                                var["coord_lower_left"]["y"] = vertical_counter
                                var["coord_upper_left"]["x"] = horizontal_counter
                                # upper counter "y" coordinate remains the same
                                # compute the area and compare it accordignly
                                a = height - var["coord_upper_left"]["y"]
                                if a > area:
                                    area = a
                                    ymax = height - var["coord_upper_left"]["y"]
                                    ymin = height - var["coord_lower_left"]["y"] - 1
                                    xmax = horizontal_counter + 1
                                    xmin = var["coord_upper_left"]["x"]
                                # Now we compute areas of above unhindered rectangles
                                # and compare their areas
                                length_unhindered = len(var["unhindered"])
                                for l in range(length_unhindered):
                                    unhindered_area = compute_area(
                                        (height - var["unhindered"][l][2]),
                                        (height - var["unhindered"][l][0] - 1),
                                        (horizontal_counter + 1),
                                        var["unhindered"][l][1],
                                    )
                                    if unhindered_area > area:
                                        area = unhindered_area
                                        ymax = height - var["unhindered"][l][2]
                                        ymin = height - var["unhindered"][l][0] - 1
                                        xmax = horizontal_counter + 1
                                        xmin = var["unhindered"][l][1]
                                break
                            # We remained higher than our lower left bound that we had set
                            # we therefore get rid of unhindered elements (rectangles) that do not need
                            # to be considered because they have become hindered
                            if var["coord_lower_left"]["y"] > (vertical_counter):
                                # first insert a new unhindered element lower bound between current hindered elements
                                length_unhindered = len(var["unhindered"])
                                for l in range(length_unhindered):
                                    if var["unhindered"][l][0] > vertical_counter - 1:
                                        var["unhindered"].insert(
                                            l,
                                            [
                                                vertical_counter - 1,
                                                var["unhindered"][l][1],
                                                var["unhindered"][l][2],
                                                var["unhindered"][l][3],
                                            ],
                                        )
                                        break
                                # now get rid of hindered elements
                                length_unhindered = len(var["unhindered"])
                                if length_unhindered != 0:
                                    indices = []
                                    for l in range(length_unhindered):
                                        if var["unhindered"][l][0] > (
                                            vertical_counter - 1
                                        ):
                                            indices.append(l)
                                    indices.reverse()
                                    for indice in indices:
                                        var["unhindered"].pop(indice)
                                # compute remaining areas
                                # first check to see if there were indeed unhidered elements previously created
                                # compute and compare their areas
                                if length_unhindered != 0:
                                    length_unhindered = len(var["unhindered"])
                                    var["coord_lower_left"]["y"] = vertical_counter - 1
                                    var["coord_lower_left"]["x"] = (
                                        var["unhindered"][length_unhindered - 1][1] + 1
                                    )
                                    for l in range(length_unhindered):
                                        unhindered_area = compute_area(
                                            (height - var["unhindered"][l][2]),
                                            (height - var["unhindered"][l][0] - 1),
                                            (horizontal_counter + 1),
                                            var["unhindered"][l][1],
                                        )
                                        if unhindered_area > area:
                                            area = unhindered_area
                                            ymax = height - var["unhindered"][l][2]
                                            ymin = height - var["unhindered"][l][0] - 1
                                            xmax = horizontal_counter + 1
                                            xmin = var["unhindered"][l][1]

                                if length_unhindered == 0:
                                    var["coord_lower_left"]["y"] = vertical_counter - 1
                                    # compute one area
                                    a = compute_area(
                                        (height - var["coord_upper_left"]["y"]),
                                        (height - var["coord_lower_left"]["y"] - 1),
                                        (horizontal_counter + 1),
                                        (var["coord_lower_left"]["x"]),
                                    )
                                    if a > area:
                                        area = a
                                        ymax = height - var["coord_upper_left"]["y"]
                                        ymin = height - var["coord_lower_left"]["y"] - 1
                                        xmax = horizontal_counter + 1
                                        xmin = var["coord_lower_left"]["x"]
                                break
                            # if we stay at the same lower bound
                            if (
                                var["coord_lower_left"]["y"] == (vertical_counter - 1)
                            ) or (
                                var["coord_lower_left"]["y"] == vertical_counter
                                and vertical_counter == height - 1
                            ):
                                # compute and compare
                                a = compute_area(
                                    (height - var["coord_upper_left"]["y"]),
                                    (height - var["coord_lower_left"]["y"] - 1),
                                    (horizontal_counter + 1),
                                    (var["coord_lower_left"]["x"]),
                                )

                                if a > area:
                                    area = a
                                    ymax = height - var["coord_upper_left"]["y"]
                                    ymin = height - var["coord_lower_left"]["y"] - 1
                                    xmax = horizontal_counter + 1
                                    xmin = var["coord_lower_left"]["x"]
                                # check unhindered elements
                                length_unhindered = len(var["unhindered"])
                                for l in range(length_unhindered):
                                    unhindered_area = compute_area(
                                        (height - var["unhindered"][l][2]),
                                        (height - var["unhindered"][l][0] - 1),
                                        (horizontal_counter + 1),
                                        var["unhindered"][l][1],
                                    )
                                    if unhindered_area > area:
                                        area = unhindered_area
                                        ymax = height - var["unhindered"][l][2]
                                        ymin = height - var["unhindered"][l][0] - 1
                                        xmax = horizontal_counter + 1
                                        xmin = var["unhindered"][l][1]
                                break
                            break
                reset_variables(var)

    results = [xmin, (height - ymax), (xmax - 1), (height - ymin - 1)]

    return results
