"""Force dataset base class for kgcnn-torch.

Ported from kgcnn.data.force — extends QMDataset with force coordinate loading.
"""
import os
import numpy as np
from kgcnn_torch.molecule.base import MolGraphInterface
from typing import Union, Callable, Dict
from kgcnn_torch.data.qm import QMDataset
from kgcnn_torch.molecule.io import parse_list_to_xyz_str, read_xyz_file, write_list_to_xyz_file


class ForceDataset(QMDataset):
    r"""Base class for Force datasets. Inherits from QMDataset.

    Extends QMDataset with the ability to read force xyz-files.

    .. code-block:: console

        data_directory/
            file_directory/
                *.xyz
            file_name.csv
            file_name.xyz
            file_name.sdf
            file_name_force.xyz
            dataset_name.kgcnn.pickle
    """

    def __init__(self, data_directory: str = None, dataset_name: str = None, file_name: str = None,
                 verbose: int = 10, file_directory: str = None,
                 file_name_xyz: str = None,
                 file_name_mol: str = None,
                 file_name_force_xyz: str = None):
        super(ForceDataset, self).__init__(data_directory=data_directory, dataset_name=dataset_name,
                                           file_name=file_name, verbose=verbose, file_directory=file_directory)
        self.label_units = None
        self.label_names = None
        self.file_name_xyz = file_name_xyz
        self.file_name_mol = file_name_mol
        self.file_name_force_xyz = file_name_force_xyz

    @property
    def file_path_force_xyz(self):
        """File path for force xyz-file(s)."""
        self._verify_data_directory()
        if self.file_name_force_xyz is None:
            return os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + "_force.xyz")
        elif isinstance(self.file_name_force_xyz, (str, os.PathLike)):
            return os.path.join(self.data_directory, self.file_name_force_xyz)
        elif isinstance(self.file_name_force_xyz, (list, tuple)):
            return [os.path.join(self.data_directory, x) for x in self.file_name_force_xyz]
        else:
            raise TypeError("Wrong type for `file_name_force_xyz` : '%s'." % self.file_name_force_xyz)

    def prepare_data(self, overwrite: bool = False, file_column_name: str = None, file_column_name_force: str = None,
                     make_sdf: bool = False):
        r"""Prepare data including force coordinate files.

        Args:
            overwrite (bool): Overwrite existing files. Default is False.
            file_column_name (str): Column name for position xyz files.
            file_column_name_force (str, list): Column name(s) for force xyz files.
            make_sdf (bool): Whether to make SDF file. Default is False.

        Returns:
            self
        """
        super(ForceDataset, self).prepare_data(overwrite=overwrite, file_column_name=file_column_name,
                                               make_sdf=make_sdf)

        file_path_forces = self.file_path_force_xyz
        if not isinstance(file_path_forces, (list, tuple)):
            file_path_forces = [file_path_forces]
        if not isinstance(file_column_name_force, (list, tuple)):
            file_column_name_force = [file_column_name_force]
        for f, c in zip(file_path_forces, file_column_name_force):
            if not os.path.exists(f):
                xyz_list = self.collect_files_in_file_directory(
                    file_column_name=c, table_file_path=None,
                    read_method_file=self.get_geom_from_xyz_file, update_counter=self._default_loop_update_info,
                    append_file_content=True, read_method_return_list=True
                )
                write_list_to_xyz_file(f, xyz_list)

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
        """Read geometric information plus force coordinates into memory.

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
        super(ForceDataset, self).read_in_memory(
            label_column_name=label_column_name, nodes=nodes, edges=edges, graph=graph, encoder_nodes=encoder_nodes,
            encoder_edges=encoder_edges, encoder_graph=encoder_graph, add_hydrogen=add_hydrogen, sanitize=sanitize,
            make_directed=make_directed, compute_partial_charges=compute_partial_charges,
            additional_callbacks=additional_callbacks, custom_transform=custom_transform
        )
        file_path_forces = self.file_path_force_xyz
        if not isinstance(file_path_forces, (list, tuple)):
            file_path_forces = [file_path_forces]
        for i, x in enumerate(file_path_forces):
            self.read_in_memory_xyz(x, atomic_coordinates=str(os.path.basename(x)),
                                    atomic_number=None, atomic_symbol=None)
        return self
