import matplotlib.pyplot as plt

def create_canvas(**kwargs):
    """
    
    """
    
    if "fig" in kwargs.keys():
        fig = kwargs["fig"]
        ax = fig.gca()
    elif "ax" in kwargs.keys():
        ax = kwargs["ax"]
        fig = ax.figure
    else:
        fig, ax = plt.subplots()
    
    kwargs.pop("fig", None)
    kwargs.pop("ax", None)

    return fig, ax, kwargs