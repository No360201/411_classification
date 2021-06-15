from torchvision import transforms

def transform_train(size):
    transform_data = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(size),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                            std=[0.5, 0.5, 0.5])])
    return transform_data

def transform_test(size):
    transform_data_test = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                               std=[0.5, 0.5, 0.5])])
    return transform_data_test