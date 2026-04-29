import torch

def unpack_cube(cube: torch.Tensor)-> tuple[
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor
]:
    """
    Unpack cubed targets/predictions into the respective components.

    NOTE: function assumes there are 7 channels in the output, per grid
    cell. Each structured like this:
    0: center x of bounding box, relative to grid cell (0-1)
    1: center y of bounding box, relative to grid cell (0-1)
    2: bounding box width, relative to full image (0-1)
    3: bounding box height, relative to full image (0-1)
    4: confidence there is an object
    5, 6: class confidence scores.

    :returns: In order (shape=(batch, `grid_size`, `grid_size`)): 
        x component,
        y component,
        width component,
        height component,
        object confidence component,
        class label components,
    :rtype: tuple[
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor
    ]
    """
    x = cube[..., 0]
    y = cube[..., 1]
    w = cube[..., 2]
    h = cube[..., 3]
    objectness = cube[..., 4]
    classes = cube[..., 5:] 
    return x, y, w, h, objectness, classes

def decode_predictions(
    output: torch.Tensor, 
    grid_size: int | None=None
)-> tuple[
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor
]:
    """
    Decode the predictions into the respective components.

    NOTE: function assumes there are 7 channels in the output, per grid
    cell. Each structured like this:
    0: center x of bounding box, relative to grid cell (0-1)
    1: center y of bounding box, relative to grid cell (0-1)
    2: bounding box width, relative to full image (0-1)
    3: bounding box height, relative to full image (0-1)
    4: confidence there is an object
    5, 6: class confidence scores. 

    :param output: Model output (shape: batch, 343) or 
        (batch, grid_size, grid_size, 7)
    :type output: torch.tensor
    :param grid_size: The size of the grid the image was cut up into.
        Each cell in this grid can contain 1 bounding box. Only needs
        to be provided, if the `output` is not yet a cube.
    :type grid_size: int | None
    :returns: In order (shape=(batch, `grid_size`, `grid_size`)): 
        center x coordinates, relative to image (0-1),
        center y coordinates, relative to image (0-1),
        bounding box widths, relative to image (0-1),
        bounding box heights, relative to image (0-1),
        object confidences,
        class confidence scores (
            shape=(batch, `grid_size`, `grid_size`, n_classes)
        )
    :rtype: tuple[
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor
    ]
    """
    if len(output.shape) > 2:
        cube_output = output
    else:
        assert grid_size is not None, \
            "Provided output was not yet a cube, " \
            "yet no `grid_size` was provided."
        cube_output = output.view(-1, grid_size, grid_size, 7)

    x, y, w, h, object_confidence, classes = unpack_cube(cube_output)

    # x, y centre data is still relative to the respective grid cell,
    # which we have to 'normalise' using offsets.
    cell_indexes = torch.arange(7, device=output.device)
    column_offsets = cell_indexes.view(1, 1, 7)
    row_offsets = cell_indexes.view(1, 7, 1)

    corrected_x = (column_offsets + x) / 7
    corrected_y = (row_offsets + y) / 7

    return corrected_x, corrected_y, w, h, object_confidence, classes  
