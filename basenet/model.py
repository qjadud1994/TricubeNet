
def Model_factory(backbone, num_classes):
    if backbone == 'hourglass52':
        from basenet.hourglass import StackedHourGlass as Model
        model = Model(num_classes, 1)
        
    elif backbone == 'hourglass104':
        from basenet.hourglass import StackedHourGlass as Model
        model = Model(num_classes, 2)
        
    elif backbone == 'hourglass104_MRCB':
        from basenet.hourglass_MRCB import StackedHourGlass as Model
        model = Model(num_classes, 2)
        
    elif backbone == 'hourglass104_MRCB_cascade':
        from basenet.hourglass_MRCB_cascade import StackedHourGlass as Model
        model = Model(num_classes, 2)

    elif backbone == 'uesnet101_dcn':
        from basenet.resnet_dcn import get_uesnet101 as Model
        model = Model(num_classes)
        
    elif backbone == 'uesnet18_dcn':
        from basenet.resnet_dcn import get_uesnet18 as Model
        model = Model(num_classes)
        
    elif backbone == 'DLA_dcn':
        from basenet.DLA_dcn import get_pose_net as Model
        model = Model(num_classes)
        
    elif backbone == 'hhrnet32':
        from basenet.higher_HRNet import Higher_HRNet32
        model = Higher_HRNet32(num_classes)
        
    elif backbone == 'hhrnet48':
        from basenet.higher_HRNet import Higher_HRNet48
        model = Higher_HRNet48(num_classes)
        
    else:
        raise "Model import Error !! "
        
    return model

