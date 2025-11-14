# Feature_Controlled_Cubic_Bezier_Curves
Paper: Efficient Construction of Feature-Controlled Cubic Bézier Curves with Continuity.

# Dependency installation

```pip install numpy matplotlib tkinter```

# Running Steps
1. Download the code and unzip it to a local directory.
2. Enter the code directory and run the main program directly:

``` python Feature_Controlled_Cubic_Bezier_Curves.py```

or use PyCharm

# Interface Features and Usage Guide
## Left parameter control panel
### 1. Model Management

Models：Select the loaded model file.

load model：Load the selected model and draw on the canvas.

Exit model：Exit model mode and return to mouse-drawn customization.

### 2. Curve Type

closed curve：Create a closed curve.

open curve：Create an open curve.

### 3. Display options 

show the control polygon：Show the control polygon for the curve.

show the curvature：Show the curvature comb. The model does not have a loadable curvature comb.

only show the curve：Display only the final curve and hide feature points and control polygons.

### 4. Feature point

The default drawing is a Regular point. Right-click to change the feature point type.

Select the type of feature point for the curve, including：

Cusp(a=2), Inflection point, Loop, Regular point

The article provides detailed formula derivations for four types of feature points.

### 5. Set parameter a

Inflection point value (1 < a < 2)：Adjust the inflection point parameter a, with the slider range 1.01 to 1.99. default value：1.1.

- Inflection point symbol：Sub or Add, two cases of inflection points in the article, you can select symbols based on the drawn shapes.

Loop value (a > 2)：Adjust the parameter a of the pivot point. The slider ranges from 2.01 to 10.0. default value：3.0.

Regular point (0 < a < 1)：Adjust the regular point parameter a by sliding the slider from 0.1 to 0.99. default value：0.67.

Cusp: Fixed value: 2, no setting required.

### 6. Curved comb settings

scale：Adjust the zoom factor of the curvature comb (to control its length).

density：Adjust the density of the curvature comb (control the number of combs).

### 7. other

Clear all：Clear all custom-drawn points and curves on the canvas.

Reset the view：Reset the canvas view.

## Right drawing area

Left-click：Add a new feature point (default is Regular point). If you click an existing feature point, you can drag it to adjust its position.

Right-click：Select feature points. After selecting them, you can adjust their type and parameters in the left panel. Right-click an empty area to deselect.
