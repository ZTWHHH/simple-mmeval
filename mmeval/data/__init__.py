from .local import LocalJSONDataset

def load_dataset(args):
    if args.dataset == "local@json":
        return LocalJSONDataset(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")