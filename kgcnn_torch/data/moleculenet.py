"""MoleculeNet dataset base class for kgcnn-torch.

Ported from kgcnn.data.moleculenet — loads SMILES-based molecular datasets.
Includes the map_molecule_callbacks() helper function.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Callable, Union, List
from collections import defaultdict
from kgcnn_torch.molecule.serial import deserialize_encoder
from kgcnn_torch.data.base import MemoryGraphDataset
from kgcnn_torch.molecule.base import MolGraphInterface
from kgcnn_torch.molecule.encoder import OneHotEncoder
from kgcnn_torch.molecule.io import write_mol_block_list_to_sdf, read_mol_list_from_sdf_file, write_smiles_file
from kgcnn_torch.molecule.convert import MolConverter

try:
    from kgcnn_torch.molecule.graph_rdkit import MolecularGraphRDKit
except ModuleNotFoundError:
    MolecularGraphRDKit = None


def map_molecule_callbacks(mol_list: List[str],
                           data: Union[pd.Series, pd.DataFrame],
                           callbacks: Dict[str, Callable[[MolGraphInterface, pd.Series], None]],
                           custom_transform: Callable[[MolGraphInterface], MolGraphInterface] = None,
                           add_hydrogen: bool = False,
                           make_directed: bool = False,
                           sanitize: bool = True,
                           compute_partial_charges: str = None,
                           mol_interface_class=None,
                           logger=None,
                           loop_update_info: int = 5000
                           ) -> dict:
    r"""Iterate over molecules, creating MolGraphInterface for each, and invoke callbacks.

    Args:
        mol_list (list): List of mol-block strings.
        data (pd.DataFrame): Pandas data frame matching the molecule list.
        callbacks (dict): Dict of callbacks {name: fn(mg, ds) -> value}.
        custom_transform (Callable): Custom transformation on MolGraphInterface.
        add_hydrogen (bool): Whether to add hydrogen. Default is False.
        make_directed (bool): Whether to have directed bonds. Default is False.
        sanitize (bool): Whether to sanitize molecule. Default is True.
        compute_partial_charges (str): Compute partial charges method. Default is None.
        mol_interface_class: MolGraphInterface class to use.
        logger: Logger instance.
        loop_update_info (int): Progress update interval.

    Returns:
        dict: Values of callbacks, each a list of length len(mol_list).
    """
    if data is None:
        if logger is not None:
            logger.error("Received no pandas data.")
    if mol_list is None:
        raise ValueError("Expected list of mol-string. But got '%s'." % mol_list)

    value_lists = defaultdict(list)
    for index, sm in enumerate(mol_list):

        mg = mol_interface_class(make_directed=make_directed).from_mol_block(
            sm, keep_hs=add_hydrogen, sanitize=sanitize)

        if custom_transform is not None:
            mg = custom_transform(mg)

        if compute_partial_charges:
            mg.compute_partial_charges(method=compute_partial_charges)

        for name, callback in callbacks.items():
            if mg.mol is None:
                value_lists[name].append(None)
            else:
                if data is not None:
                    data_dict = data.loc[index]
                else:
                    data_dict = None
                value = callback(mg, data_dict)
                value_lists[name].append(value)
        if index % loop_update_info == 0:
            if logger is not None:
                logger.info(" ... process molecules {0} from {1}".format(index, len(mol_list)))

    return value_lists


class MoleculeNetDataset(MemoryGraphDataset):
    r"""Class for using MoleculeNet-style datasets.

    Loads SMILES from CSV, converts to molecular graphs via RDKit, and extracts features.

    .. code-block:: console

        data_directory/
            file_name.csv
            file_name.SMILES
            file_name.sdf
            dataset_name.kgcnn.pickle
    """

    _default_node_attributes = [
        'Symbol', 'TotalDegree', 'FormalCharge', 'NumRadicalElectrons', 'Hybridization',
        'IsAromatic', 'IsInRing', 'TotalNumHs', 'CIPCode', "ChiralityPossible", "ChiralTag"
    ]
    _default_node_encoders = {
        'Symbol': OneHotEncoder(
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
            dtype="str"
        ),
        'Hybridization': OneHotEncoder([2, 3, 4, 5, 6]),
        'TotalDegree': OneHotEncoder([0, 1, 2, 3, 4, 5], add_unknown=False),
        'TotalNumHs': OneHotEncoder([0, 1, 2, 3, 4], add_unknown=False),
        'CIPCode': OneHotEncoder(['R', 'S'], add_unknown=False, dtype='str'),
        "ChiralityPossible": OneHotEncoder(["1"], add_unknown=False, dtype='str'),
    }
    _default_edge_attributes = ['BondType', 'IsAromatic', 'IsConjugated', 'IsInRing', 'Stereo']
    _default_edge_encoders = {
        'BondType': OneHotEncoder([1, 2, 3, 12], add_unknown=False),
        'Stereo': OneHotEncoder([0, 1, 2, 3], add_unknown=False)
    }
    _default_graph_attributes = ['ExactMolWt', 'NumAtoms']
    _default_graph_encoders = {}

    _default_loop_update_info = 5000
    _mol_graph_interface = MolecularGraphRDKit

    def __init__(self, data_directory: str = None, dataset_name: str = None, file_name: str = None,
                 file_name_mol: str = None, file_name_smiles: str = None, verbose: int = 10):
        MemoryGraphDataset.__init__(self, data_directory=data_directory, dataset_name=dataset_name,
                                    file_name=file_name, verbose=verbose)
        self.file_name_mol = file_name_mol
        self.file_name_smiles = file_name_smiles

    @property
    def file_path_mol(self):
        """File path for the SDF mol information."""
        self._verify_data_directory()
        if self.file_name_mol is None:
            return os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + ".sdf")
        else:
            return os.path.join(self.data_directory, self.file_name_mol)

    @property
    def file_path_smiles(self):
        """File path for the SMILES file."""
        self._verify_data_directory()
        if self.file_name_smiles is None:
            return os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + ".SMILES")
        else:
            return os.path.join(self.data_directory, self.file_name_smiles)

    def prepare_data(self, overwrite: bool = False, smiles_column_name: str = "smiles",
                     add_hydrogen: bool = True, sanitize: bool = True,
                     make_conformers: bool = True, optimize_conformer: bool = True,
                     external_program: dict = None, num_workers: int = None):
        r"""Generate molecular structures from SMILES and store as SDF.

        Args:
            overwrite (bool): Overwrite existing SDF. Default is False.
            smiles_column_name (str): Column name for SMILES in CSV file. Default is "smiles".
            add_hydrogen (bool): Add hydrogen after SMILES translation. Default is True.
            sanitize (bool): Sanitize molecule. Default is True.
            make_conformers (bool): Make 3D conformers. Default is True.
            optimize_conformer (bool): Optimize conformer via force field. Default is True.
            external_program (dict): External program config for SMILES translation.
            num_workers (int): Parallel workers for SMILES translation.

        Returns:
            self
        """
        if os.path.exists(self.file_path_mol) and not overwrite:
            self.info("Found SDF %s of pre-computed structures." % self.file_path_mol)
            return self

        self.read_in_table_file()
        smiles = self.data_frame[smiles_column_name].values
        if len(smiles) == 0:
            self.error("Can not translate smiles, received empty list for '%s'." % self.dataset_name)
        write_smiles_file(self.file_path_smiles, smiles)

        self.info("Generating molecules and store %s to disk..." % self.file_path_mol)
        conv = MolConverter()
        conv.smile_to_mol(
            self.file_path_smiles, self.file_path_mol, add_hydrogen=add_hydrogen, sanitize=sanitize,
            make_conformers=make_conformers, optimize_conformer=optimize_conformer,
            external_program=external_program, num_workers=num_workers,
            logger=self.logger, batch_size=self._default_loop_update_info
        )
        return self

    def get_mol_blocks_from_sdf_file(self):
        if not os.path.exists(self.file_path_mol):
            raise FileNotFoundError("Can not load molecules for dataset %s" % self.dataset_name)
        self.info("Read molecules from mol-file.")
        return read_mol_list_from_sdf_file(self.file_path_mol)

    def set_attributes(self,
                       label_column_name: Union[str, list] = None,
                       nodes: list = None,
                       edges: list = None,
                       graph: list = None,
                       encoder_nodes: dict = None,
                       encoder_edges: dict = None,
                       encoder_graph: dict = None,
                       add_hydrogen: bool = False,
                       make_directed: bool = False,
                       has_conformers: bool = True,
                       sanitize: bool = True,
                       compute_partial_charges: str = None,
                       additional_callbacks: Dict[str, Callable[[MolGraphInterface, dict], None]] = None,
                       custom_transform: Callable[[MolGraphInterface], MolGraphInterface] = None):
        """Load molecules from SDF into memory and extract features.

        Args:
            label_column_name (str, list): Column name(s) for graph labels in CSV.
            nodes (list): Node attributes to extract.
            edges (list): Edge attributes to extract.
            graph (list): Graph attributes to extract.
            encoder_nodes (dict): Encoders for node attributes.
            encoder_edges (dict): Encoders for edge attributes.
            encoder_graph (dict): Encoders for graph attributes.
            add_hydrogen (bool): Keep hydrogen. Default is False.
            make_directed (bool): Directed bonds. Default is False.
            has_conformers (bool): Add node coordinates. Default is True.
            sanitize (bool): Sanitize molecule. Default is True.
            compute_partial_charges (str): Compute partial charges. Default is None.
            additional_callbacks (dict): Additional custom callbacks.
            custom_transform (Callable): Custom transformation function.

        Returns:
            self
        """
        nodes = nodes if nodes is not None else self._default_node_attributes
        edges = edges if edges is not None else self._default_edge_attributes
        graph = graph if graph is not None else self._default_graph_attributes
        encoder_nodes = encoder_nodes if encoder_nodes is not None else self._default_node_encoders
        encoder_edges = encoder_edges if encoder_edges is not None else self._default_edge_encoders
        encoder_graph = encoder_graph if encoder_graph is not None else self._default_graph_encoders
        additional_callbacks = additional_callbacks if additional_callbacks is not None else {}

        for encoder in [encoder_nodes, encoder_edges, encoder_graph]:
            for key, value in encoder.items():
                encoder[key] = deserialize_encoder(value)

        callbacks = {
            'node_symbol': lambda mg, ds: mg.node_symbol,
            'node_number': lambda mg, ds: mg.node_number,
            'edge_indices': lambda mg, ds: mg.edge_number[0],
            'edge_number': lambda mg, ds: np.array(mg.edge_number[1], dtype='int'),
            'graph_size': lambda mg, ds: len(mg.node_number),
        }
        if has_conformers:
            callbacks.update({'node_coordinates': lambda mg, ds: mg.node_coordinates})
        if label_column_name:
            callbacks.update({'graph_labels': lambda mg, ds: ds[label_column_name]})

        callbacks.update({
            'node_attributes': lambda mg, ds: np.array(mg.node_attributes(nodes, encoder_nodes), dtype='float32'),
            'edge_attributes': lambda mg, ds: np.array(mg.edge_attributes(edges, encoder_edges)[1], dtype='float32'),
            'graph_attributes': lambda mg, ds: np.array(mg.graph_attributes(graph, encoder_graph), dtype='float32')
        })

        callbacks.update(additional_callbacks)

        value_lists = map_molecule_callbacks(
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

        for name, values in value_lists.items():
            self.assign_property(name, values)

        if self.logger.getEffectiveLevel() < 20:
            for encoder in [encoder_nodes, encoder_edges, encoder_graph]:
                for key, value in encoder.items():
                    if hasattr(value, "report"):
                        value.report(name=key)

        return self

    read_in_memory = set_attributes
