"""QM (Quantum Mechanics) dataset base class for kgcnn-torch.

Ported from kgcnn.data.qm — generates graph properties from xyz-files.
"""
import os
import numpy as np
from typing import Union, Callable, Dict
from kgcnn_torch.molecule.base import MolGraphInterface
from kgcnn_torch.data.transform import QMGraphLabelScaler
from kgcnn_torch.molecule.serial import deserialize_encoder
from kgcnn_torch.data.base import MemoryGraphDataset
from kgcnn_torch.molecule.io import parse_list_to_xyz_str, read_xyz_file, \
    write_mol_block_list_to_sdf, read_mol_list_from_sdf_file, write_list_to_xyz_file
from kgcnn_torch.molecule.methods import global_proton_dict, inverse_global_proton_dict
from kgcnn_torch.molecule.convert import MolConverter
from kgcnn_torch.data.moleculenet import map_molecule_callbacks

try:
    from kgcnn_torch.molecule.graph_babel import MolecularGraphOpenBabel
except ModuleNotFoundError:
    MolecularGraphOpenBabel = None

try:
    from kgcnn_torch.molecule.graph_rdkit import MolecularGraphRDKit
except ModuleNotFoundError:
    MolecularGraphRDKit = None


class QMDataset(MemoryGraphDataset):
    r"""Base class for QM (quantum mechanical) datasets.

    Generates graph properties from xyz-files storing atomic coordinates.
    Loading multiple single xyz-files into one file is supported.
    File names and labels are given by a CSV or table file.

    .. code-block:: console

        data_directory/
            file_directory/
                *.xyz
            file_name.csv
            file_name.xyz
            file_name.sdf
            dataset_name.kgcnn.pickle
    """

    _global_proton_dict = global_proton_dict
    _inverse_global_proton_dict = inverse_global_proton_dict
    _default_loop_update_info = 5000
    _mol_graph_interface = MolecularGraphRDKit

    def __init__(self, data_directory: str = None, dataset_name: str = None, file_name: str = None,
                 verbose: int = 10, file_directory: str = None, file_name_xyz: str = None, file_name_mol: str = None):
        MemoryGraphDataset.__init__(self, data_directory=data_directory, dataset_name=dataset_name,
                                    file_name=file_name, verbose=verbose,
                                    file_directory=file_directory)
        self.label_units = None
        self.label_names = None
        self.file_name_xyz = file_name_xyz
        self.file_name_mol = file_name_mol

    @property
    def file_path_mol(self):
        """File path for the SDF mol information."""
        self._verify_data_directory()
        if self.file_name_mol is None:
            return os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + ".sdf")
        else:
            return os.path.join(self.data_directory, self.file_name_mol)

    @property
    def file_path_xyz(self):
        """File path for the XYZ coordinate file."""
        self._verify_data_directory()
        if self.file_name_xyz is None:
            return os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + ".xyz")
        else:
            return os.path.join(self.data_directory, self.file_name_xyz)

    def get_geom_from_xyz_file(self, file_path: str) -> list:
        """Get a list of xyz items from file."""
        if file_path is None:
            file_path = self.file_path_xyz
        return read_xyz_file(file_path)

    def get_mol_blocks_from_sdf_file(self, file_path: str = None) -> list:
        """Get a list of mol-blocks from SDF file."""
        if file_path is None:
            file_path = self.file_path_mol
        if not os.path.exists(file_path):
            raise FileNotFoundError("Can not load SDF for dataset %s" % self.dataset_name)
        mol_list = read_mol_list_from_sdf_file(file_path)
        if mol_list is None:
            self.warning("Failed to load bond information from SDF file.")
        return mol_list

    def prepare_data(self, overwrite: bool = False, file_column_name: str = None, make_sdf: bool = True):
        r"""Pre-computation of molecular structure information in a sdf-file from a xyz-file.

        Args:
            overwrite (bool): Overwrite existing SDF file. Default is False.
            file_column_name (str): Name of the column in csv file with xyz-file names.
            make_sdf (bool): Whether to make SDF from xyz via RDKit/OpenBabel. Default is True.

        Returns:
            self
        """
        if os.path.exists(self.file_path_mol) and not overwrite:
            self.info("Found SDF file '%s' of pre-computed structures." % self.file_path_mol)
            return self

        if not os.path.exists(self.file_path_xyz):
            xyz_list = self.collect_files_in_file_directory(
                file_column_name=file_column_name, table_file_path=None,
                read_method_file=self.get_geom_from_xyz_file, update_counter=self._default_loop_update_info,
                append_file_content=True, read_method_return_list=True
            )
            write_list_to_xyz_file(self.file_path_xyz, xyz_list)

        if make_sdf:
            self.info("Converting xyz to mol information.")
            converter = MolConverter()
            converter.xyz_to_mol(self.file_path_xyz, self.file_path_mol)
        return self

    def read_in_memory_xyz(self, file_path: str = None,
                           atomic_coordinates: Union[str, None] = "node_coordinates",
                           atomic_symbol: Union[str, None] = "node_symbol",
                           atomic_number: Union[str, None] = "node_number"):
        """Read XYZ-file with geometric information into memory.

        Args:
            file_path (str): Filepath to xyz file.
            atomic_coordinates (str): Name of graph property for coordinates. Default is "node_coordinates".
            atomic_symbol (str): Name of graph property for symbol. Default is "node_symbol".
            atomic_number (str): Name of graph property for atomic number. Default is "node_number".

        Returns:
            self
        """
        xyz_list = self.get_geom_from_xyz_file(file_path)
        symbol = [np.array(x[0]) for x in xyz_list]
        coord = [np.array(x[1], dtype="float")[:, :3] for x in xyz_list]
        nodes = [np.array([self._global_proton_dict[x] for x in y[0]], dtype="int") for y in xyz_list]
        for key, value in zip([atomic_coordinates, atomic_symbol, atomic_number], [coord, symbol, nodes]):
            if key is not None:
                self.assign_property(key, value)
        return self

    def set_attributes(self,
                       label_column_name: Union[str, list] = None,
                       nodes: list = None,
                       edges: list = None,
                       graph: list = None,
                       encoder_nodes: dict = None,
                       encoder_edges: dict = None,
                       encoder_graph: dict = None,
                       add_hydrogen: bool = True,
                       make_directed: bool = False,
                       sanitize: bool = False,
                       compute_partial_charges: str = None,
                       additional_callbacks: Dict[str, Callable[[MolGraphInterface, dict], None]] = None,
                       custom_transform: Callable[[MolGraphInterface], MolGraphInterface] = None):
        """Read SDF-file with chemical structure information into memory.

        Args:
            label_column_name (str, list): Name of labels for columns in CSV file.
            nodes (list): List of node attributes as string.
            edges (list): List of edge attributes as string.
            graph (list): List of graph attributes as string.
            encoder_nodes (dict): Dict of callable encoders for node attributes.
            encoder_edges (dict): Dict of callable encoders for edge attributes.
            encoder_graph (dict): Dict of callable encoders for graph attributes.
            add_hydrogen (bool): Whether to keep hydrogen. Default is True.
            make_directed (bool): Whether to have directed bonds. Default is False.
            sanitize (bool): Whether to sanitize molecule. Default is False.
            compute_partial_charges (str): Compute partial charges method. Default is None.
            additional_callbacks (dict): Additional callbacks for custom properties.
            custom_transform (Callable): Custom transformation function.

        Returns:
            self
        """
        additional_callbacks = additional_callbacks if additional_callbacks is not None else {}

        for encoder in [encoder_nodes, encoder_edges, encoder_graph]:
            if encoder is not None:
                for key, value in encoder.items():
                    encoder[key] = deserialize_encoder(value)

        callbacks = {
            "node_symbol": lambda mg, ds: mg.node_symbol,
            "node_number": lambda mg, ds: mg.node_number,
            "node_coordinates": lambda mg, ds: mg.node_coordinates,
            "edge_indices": lambda mg, ds: mg.edge_number[0],
            "edge_number": lambda mg, ds: np.array(mg.edge_number[1], dtype='int'),
            **additional_callbacks
        }
        if label_column_name:
            callbacks.update({'graph_labels': lambda mg, ds: ds[label_column_name]})

        if nodes:
            callbacks.update({
                'node_attributes': lambda mg, ds: np.array(mg.node_attributes(nodes, encoder_nodes), dtype='float32')
            })
        if edges:
            callbacks.update({
                'edge_attributes': lambda mg, ds: np.array(mg.edge_attributes(edges, encoder_edges)[1], dtype='float32')
            })
        if graph:
            callbacks.update({
                'graph_attributes': lambda mg, ds: np.array(mg.graph_attributes(graph, encoder_graph), dtype='float32')
            })

        value_list = map_molecule_callbacks(
            self.get_mol_blocks_from_sdf_file(),
            self.read_in_table_file().data_frame,
            callbacks=callbacks,
            add_hydrogen=add_hydrogen,
            custom_transform=custom_transform,
            make_directed=make_directed,
            sanitize=sanitize,
            mol_interface_class=self._mol_graph_interface,
            logger=self.logger,
            loop_update_info=self._default_loop_update_info,
            compute_partial_charges=compute_partial_charges
        )

        for name, values in value_list.items():
            self.assign_property(name, values)

        return self

    def read_in_memory(self,
                       label_column_name: Union[str, list] = None,
                       nodes: list = None,
                       edges: list = None,
                       graph: list = None,
                       encoder_nodes: dict = None,
                       encoder_edges: dict = None,
                       encoder_graph: dict = None,
                       add_hydrogen: bool = True,
                       sanitize: bool = False,
                       make_directed: bool = False,
                       compute_partial_charges: bool = False,
                       additional_callbacks: Dict[str, Callable[[MolGraphInterface, dict], None]] = None,
                       custom_transform: Callable[[MolGraphInterface], MolGraphInterface] = None):
        """Read geometric information into memory. Tries SDF first, falls back to XYZ.

        Args:
            label_column_name (str, list): Name of labels for columns in CSV file.
            nodes (list): List of node attributes.
            edges (list): List of edge attributes.
            graph (list): List of graph attributes.
            encoder_nodes (dict): Dict of callable encoders.
            encoder_edges (dict): Dict of callable encoders.
            encoder_graph (dict): Dict of callable encoders.
            add_hydrogen (bool): Whether to keep hydrogen. Default is True.
            make_directed (bool): Whether to have directed bonds. Default is False.
            compute_partial_charges (str): Compute partial charges method.
            sanitize (bool): Whether to sanitize molecule. Default is False.
            additional_callbacks (dict): Additional callbacks.
            custom_transform (Callable): Custom transformation function.

        Returns:
            self
        """
        if os.path.exists(self.file_path_mol) and self._mol_graph_interface is not None:
            self.info("Reading structures from SDF file.")
            self.set_attributes(
                label_column_name=label_column_name, nodes=nodes, edges=edges, graph=graph,
                encoder_nodes=encoder_nodes, encoder_edges=encoder_edges, encoder_graph=encoder_graph,
                add_hydrogen=add_hydrogen, make_directed=make_directed, additional_callbacks=additional_callbacks,
                sanitize=sanitize, custom_transform=custom_transform
            )
        else:
            self.warning("Failed to load structures SDF file. Reading geometries from XYZ file instead.")
            data_frame = self.read_in_table_file().data_frame
            if data_frame is not None and label_column_name is not None:
                self.assign_property("graph_labels", [
                    np.array(data_frame.loc[index][label_column_name]) for index in range(data_frame.shape[0])])
            self.read_in_memory_xyz()
        return self
