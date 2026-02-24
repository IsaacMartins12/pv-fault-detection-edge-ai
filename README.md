## Data
The dataset consists of 20,000 infrared images that are 24 by 40 pixels each. There are 12 defined classes of solar modules presented in this paper with 11 classes of different anomalies and the remaining class being No-Anomaly (i.e. the null case).

|   Class Name   | Images |                                Description                                |
|:--------------:|:------:|:-------------------------------------------------------------------------:|
| Cell           | 1,877  | Hot spot occurring with square geometry in single cell.                   |
| Cell-Multi     | 1,288  | Hot spots occurring with square geometry in multiple cells.               |
| Cracking       | 941    | Module anomaly caused by cracking on module surface.                      |
| Hot-Spot       | 251    | Hot spot on a thin film module.                                           |
| Hot-Spot-Multi | 247    | Multiple hot spots on a thin film module.                                 |
| Shadowing      | 1056   | Sunlight obstructed by vegetation, man-made structures, or adjacent rows. |
| Diode          | 1,499  | Activated bypass diode, typically 1/3 of module.                          |
| Diode-Multi    | 175    | Multiple activated bypass diodes, typically affecting 2/3 of module.      |
| Vegetation     | 1,639  | Panels blocked by vegetation.                                             |
| Soiling        | 205    | Dirt, dust, or other debris on surface of module.                         |
| Offline-Module | 828    | Entire module is heated.                                                  |
| No-Anomaly     | 10,000 | Nominal solar module.                                                     |

The folder `Dataset` contains the `images` directory and `module_metadata.json` that describes each image. The JSON file is structured as follows:

```
{
  "<image_number>": {
    "image_filepath": "images/<image_number>.jpg", 
    "anomaly_class": "<class_name>"
  },
  ...
}
```

## References
This dataset was originally published at ICLR 2020 in AI for Earth Sciences workshop.

https://ai4earthscience.github.io/iclr-2020-workshop/papers/ai4earth22.pdf
