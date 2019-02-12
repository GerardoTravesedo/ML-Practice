import ConfigParser

NET_SECTION = "Net"

MODEL_SECTION = "Model"

OUTPUT_SECTION = "Output"


class ConfigReader:

    def __init__(self, prop_file):
        config = ConfigParser.ConfigParser()
        config.read(prop_file)
        self._config = config

    def get_model_load(self):
        return self._get_required_property(MODEL_SECTION, "model.load")

    def get_model_save(self):
        return self._get_required_property(MODEL_SECTION, "model.save")

    def get_number_epochs(self):
        return self._get_required_property(NET_SECTION, "number.epochs")

    def get_number_image_channels(self):
        return self._get_required_property(NET_SECTION, "number.image.channels")

    def get_number_image_pixels(self):
        return self._get_required_property(NET_SECTION, "number.image.pixels")

    def get_number_resnet_layers(self):
        return self._get_required_property(NET_SECTION, "number.resnet.layers")

    def get_number_hidden_nodes(self):
        return self._get_required_property(NET_SECTION, "number.hidden.nodes")

    def get_number_classes(self):
        return self._get_required_property(NET_SECTION, "number.classes")

    def get_number_regression_fields(self):
        return self._get_required_property(NET_SECTION, "number.regression.fields")

    def get_roi_bbox_fields(self):
        return self._get_required_property(NET_SECTION, "number.roi.bbox.fields")

    def get_logs_path(self):
        return self._get_required_property(OUTPUT_SECTION, "logs.path")

    def get_test_output_file(self):
        return self._get_required_property(OUTPUT_SECTION, "test.output.file")

    def get_training_error_file(self):
        return self._get_required_property(OUTPUT_SECTION, "training.error.file")

    def _get_required_property(self, section, property):
        property_value = self._config.get(NET_SECTION, "number.epochs")

        if not property_value:
            raise Exception("Missing property {0} in section {1}".format(property, section))

        return property_value

