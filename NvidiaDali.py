from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class HybridPipelineTrain(Pipeline):
    def __init__(self, batch_size, output_size, num_threads, device_id, images_directory):
        super(HybridPipelineTrain, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = images_directory, random_shuffle = True, initial_fill = 21)
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.rotate = ops.Rotate(device = "gpu")
        self.RRC = ops.RandomResizedCrop(size = output_size,
            random_area = [0.4, 1.0],
            random_aspect_ratio = [0.5, 1.5],
            device="gpu"
        )
        self.cmn = ops.CropMirrorNormalize(
        device="gpu",
        dtype=types.FLOAT,
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        output_layout="HWC")
        self.rng = ops.random.Uniform(range = (-25.0, 25.0))
        self.coin = ops.random.CoinFlip(probability = 0.5)
        self.flip = ops.Flip(device = "gpu")

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.flip(images, horizontal = self.coin(), vertical = self.coin())
        angle = self.rng()
        images = self.rotate(images, angle=angle)
        images = self.RRC(images)
        images = self.cmn(images)
        # images are on the GPU
        return (images, labels)

class HybridPipelineTest(Pipeline):
    def __init__(self, batch_size, output_size, num_threads, device_id, images_directory):
        super(HybridPipelineTest, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = images_directory, random_shuffle = True, initial_fill = 21)
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.cmn = ops.CropMirrorNormalize(
        device="gpu",
        dtype=types.FLOAT,
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        output_layout="HWC")
        self.res = ops.Resize(device="gpu", resize_x=output_size[0], resize_y=output_size[1], interp_type=types.INTERP_TRIANGULAR)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.res(images)
        images = self.cmn(images)
        # images are on the GPU
        return (images, labels)