"""Crystal dataset base class for kgcnn-torch.

Ported from kgcnn.data.crystal — loads crystal structures via pymatgen.
"""
import os
import numpy as np
from collections import defaultdict
from typing import Dict, Callable, List, Union
import pandas as pd
import pymatgen
import pymatgen.io.cif
import pymatgen.core.structure
import pymatgen.symmetry.structure
from kgcnn_torch.utils.serial import deserialize
from kgcnn_torch.data.base import MemoryGraphDataset, save_json_file, load_json_file
from kgcnn_torch.crystal.base import CrystalPreprocessor
from kgcnn_torch.graph.base import GraphDict


class CrystalDataset(MemoryGraphDataset):
    r"""Class for making graph dataset from periodic structures such as crystals.

    Requires a :obj:`data_directory` with a table '.csv' file containing labels
    and CIF file names in :obj:`file_directory`.

    .. code-block:: console

        data_directory/
            file_directory/
                *.cif
            file_name.csv
            file_name.pymatgen.json
            dataset_name.kgcnn.pickle

    Uses pymatgen for structure parsing and serialization.
    """

    _default_loop_update_info = 5000

    def __init__(self,
                 data_directory: str = None,
                 dataset_name: str = None,
                 file_name: str = None,
                 file_directory: str = None,
                 file_name_pymatgen_json: str = None,
                 verbose: int = 10):
        super(CrystalDataset, self).__init__(
            data_directory=data_directory, dataset_name=dataset_name, file_name=file_name, verbose=verbose,
            file_directory=file_directory)
        self._structs = None
        self.file_name_pymatgen_json = file_name_pymatgen_json
        self.label_units = None
        self.label_names = None

    @property
    def pymatgen_json_file_path(self):
        """File path for pymatgen serialization JSON."""
        self._verify_data_directory()
        if self.file_name_pymatgen_json is None:
            file_name = os.path.splitext(self.file_name)[0] + ".pymatgen.json"
        else:
            file_name = self.file_name_pymatgen_json
        return os.path.join(self.data_directory, file_name)

    @staticmethod
    def _pymatgen_serialize_structs(structs: List) -> List[dict]:
        dicts = []
        for s in structs:
            d = s.as_dict()
            dicts.append(d)
        return dicts

    @staticmethod
    def _pymatgen_deserialize_dicts(dicts: List[dict], to_unit_cell: bool = False) -> list:
        structs = []
        for x in dicts:
            s = pymatgen.core.structure.Structure.from_dict(x)
            structs.append(s)
            if to_unit_cell:
                for site in s.sites:
                    site.to_unit_cell(in_place=True)
        return structs

    def save_structures_to_json_file(self, structs: list, file_path: str = None):
        """Save a list of pymatgen structures to file."""
        if file_path is None:
            file_path = self.pymatgen_json_file_path
        self.info("Exporting as dict for pymatgen ...")
        dicts = self._pymatgen_serialize_structs(structs)
        self.info("Saving structures as .json ...")
        save_json_file(dicts, file_path)

    @staticmethod
    def _pymatgen_parse_file_to_structure(cif_file: str):
        structures = pymatgen.io.cif.CifParser(cif_file).get_structures()
        return structures

    def prepare_data(self, file_column_name: str = None, overwrite: bool = False):
        r"""Load crystal structures from CIF files and save as pymatgen JSON.

        Args:
            file_column_name (str): Column name with CIF file names. Default is None.
            overwrite (bool): Whether to rerun extraction. Default is False.

        Returns:
            self
        """
        if os.path.exists(self.pymatgen_json_file_path) and not overwrite:
            self.info("Pickled pymatgen structures already exist. Do nothing.")
            return self

        self.info("Searching for structure files in '%s'" % self.file_directory_path)
        structs = self.collect_files_in_file_directory(
            file_column_name=file_column_name, table_file_path=None,
            read_method_file=self._pymatgen_parse_file_to_structure, update_counter=self._default_loop_update_info,
            append_file_content=True, read_method_return_list=True
        )
        self.save_structures_to_json_file(structs)
        return self

    def get_structures_from_json_file(self, file_path: str = None) -> List:
        """Load pymatgen structures from JSON file.

        Args:
            file_path (str): File path to json-file. Default is None.

        Returns:
            list: List of pymatgen structures.
        """
        if file_path is None:
            file_path = self.pymatgen_json_file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError("Cannot find .json file for `CrystalDataset`. Please `prepare_data()`.")
        self.info("Reading structures from .json ...")
        return self._pymatgen_deserialize_dicts(load_json_file(file_path))

    def _map_callbacks(self, structs: list, data: pd.Series,
                       callbacks: Dict[
                           str, Callable[[pymatgen.core.structure.Structure, pd.Series], Union[np.ndarray, None]]],
                       assign_to_self: bool = True) -> dict:
        """Map callbacks on structures + data series.

        Args:
            structs (list): List of pymatgen structures.
            data (pd.Series, pd.DataFrame): Data frame matching the structures.
            callbacks (dict): Dict of callbacks {name: fn(struct, data_row) -> value}.
            assign_to_self (bool): Whether to assign results to this dataset. Default is True.

        Returns:
            dict: Values of callbacks.
        """
        value_lists = defaultdict(list)
        for index, st in enumerate(structs):
            for name, callback in callbacks.items():
                if st is None:
                    value_lists[name].append(None)
                else:
                    data_dict = data.loc[index]
                    value = callback(st, data_dict)
                    value_lists[name].append(value)
            if index % self._default_loop_update_info == 0:
                self.info(" ... read structures {0} from {1}".format(index, len(structs)))

        if assign_to_self:
            for name, values in value_lists.items():
                self.assign_property(name, values)

        return value_lists

    def read_in_memory(self, label_column_name: str = None,
                       additional_callbacks: Dict[
                           str, Callable[[pymatgen.core.structure.Structure, pd.Series], None]] = None):
        """Read structures from pymatgen JSON and convert to graph information.

        Args:
            label_column_name (str): Columns of labels. Default is None.
            additional_callbacks (dict): Additional callbacks. Default is None.

        Returns:
            self
        """
        if additional_callbacks is None:
            additional_callbacks = {}

        self.info("Making node features from structure...")
        callbacks = {
            "graph_labels": lambda st, ds: ds[label_column_name] if label_column_name is not None else None,
            "node_coordinates": lambda st, ds: np.array(st.cart_coords, dtype="float"),
            "node_frac_coordinates": lambda st, ds: np.array(st.frac_coords, dtype="float"),
            "graph_lattice": lambda st, ds: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
            "abc": lambda st, ds: np.array(st.lattice.abc),
            "charge": lambda st, ds: np.array([st.charge], dtype="float"),
            "volume": lambda st, ds: np.array([st.lattice.volume], dtype="float"),
            "node_number": lambda st, ds: np.array(st.atomic_numbers, dtype="int"),
            **additional_callbacks
        }

        self._map_callbacks(structs=self.get_structures_from_json_file(),
                            data=self.read_in_table_file(file_path=self.file_path).data_frame,
                            callbacks=callbacks)

        return self

    def set_representation(self, pre_processor: Union[CrystalPreprocessor, dict], reset_graphs: bool = False):
        r"""Build a graph representation using crystal preprocessor.

        Args:
            pre_processor (CrystalPreprocessor, dict): Crystal preprocessor to use.
            reset_graphs (bool): Whether to reset graph information. Default is False.

        Returns:
            self
        """
        if reset_graphs:
            self.clear()
        if isinstance(pre_processor, dict):
            pre_processor = deserialize(pre_processor)
        structs = self.get_structures_from_json_file()
        if reset_graphs:
            self.empty(len(structs))

        pre_processor.output_graph_as_dict = True

        for index, s in enumerate(structs):
            g = pre_processor(s)
            self[index].update(g)

            if index % self._default_loop_update_info == 0:
                self.info(" ... preprocess structures {0} from {1}".format(index, len(structs)))

        return self
