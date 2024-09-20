import base_spec
import scalar_spec
import sequence_spec
import common_spec
from blitz_spec import array_specification
blitz_type_factories = [sequence_spec.string_specification(),
                          sequence_spec.list_specification(),
                          sequence_spec.dict_specification(),
                          sequence_spec.tuple_specification(),
                          scalar_spec.int_specification(),
                          scalar_spec.float_specification(),
                          scalar_spec.complex_specification(),
                          common_spec.file_specification(),
                          common_spec.callable_specification(),
                          array_specification()]
                          #common_spec.instance_specification(),                          
                          #common_spec.module_specification()]

try: 
    # this is currently safe because it doesn't import wxPython.
