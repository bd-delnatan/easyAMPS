"""
For saving files to Prism
"""
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.dom import minidom
import numpy as np
import datetime
import pdb


class PrismFile:
    """ A simple class for generating a Prism file

    Each file may contain multiple 'Data Tables'. There is currently no way
    to automatically generate graphs for each table outside of Prism. So the
    output from this class will only contain the data that conforms to the Prism
    XML file format.

    """

    def __init__(self):
        # The Prism header file
        self.root = Element(
            "GraphPadPrismFile", attrib={"PrismXMLVersion": "5.00"}
        )

        Created = Element("Created")

        OriginalVersion = Element(
            "OriginalVersion",
            attrib={
                "CreatedByProgram": "PyBiodesy",
                "Login": "delnatan",
                "DateTime": str(datetime.datetime.now().isoformat()),
            },
        )

        MostRecentVersion = Element(
            "OriginalVersion",
            attrib={
                "CreatedByProgram": "PyBiodesy",
                "Login": "delnatan",
                "DateTime": str(datetime.datetime.now().isoformat()),
            },
        )

        Created.append(OriginalVersion)
        Created.append(MostRecentVersion)

        # add Created tag and its appended elements to root
        self.root.append(Created)

        # Write the top part
        self._add_info()

    def append(self, content):
        """ appends a PrismTable to the current file """

        # call this function to append Prism XML text for each new Tables
        self.root.append(content)

    def _add_info(self):
        # this is called in the beginning to fill in the short metadata
        Info = Element("Info", attrib={"ID": "Info0"})
        Title = SubElement(Info, "Title")
        Title.text = "Project info 1"

        SubElement(Info, "Notes")

        constant_names = [
            "Experiment Date",
            "Experiment ID",
            "Notebook ID",
            "Project",
            "Experimenter",
            "Protocol",
        ]

        now = datetime.datetime.now()
        today_yyyy_mm_dd = "{:d}-{:d}-{:d}".format(
            now.year, now.month, now.day
        )
        constant_values = [today_yyyy_mm_dd, "", "", "", "", ""]

        for (name, value) in zip(constant_names, constant_values):
            _tag = SubElement(Info, "Constant")
            _name = SubElement(_tag, "Name")
            _name.text = name
            _value = SubElement(_tag, "Value")
            _value.text = value

        self.root.append(Info)

    def _add_signature(self):
        # this is called at the end to leave the empty tag for Prism
        PrismSignature = Element(
            "Template",
            attrib={
                "dt:dt": "bin.base64",
                "xmlns:dt": "urn:schemas-microsoft-com:datatypes",
            },
        )
        # the text within the "Template" tag is just the base64 converted
        # bytes used to store images. We can leave it empty for reading files
        PrismSignature.text = ""
        self.root.append(PrismSignature)

    def write(self, output_file):
        """ write out to a PZFX file

        Args:
            output_file(str): the name of the output file, including path.
        """

        # call this method to write a prism file
        self._add_signature()
        # use ElementTree.tostring with encoding to write the XML declaration
        # at the beginning of the document
        densexmlstring = ElementTree.tostring(self.root, 'utf-8')
        nicexmlstring = minidom.parseString(densexmlstring)

        with open(output_file, "w") as fh:
            fh.write(nicexmlstring.toprettyxml(indent="    "))


def summarize_baseline_data(Plate, output_prism_file):
    """ generates prism file containing baseline signal data

    The idea is to summarize what the signals look like BEFORE adding
    any compounds.

    """

    conjugates = np.unique(Plate.Read_Plate.substance)
    Nchannels = Plate.data.Nchannels

    prism_file = PrismFile()

    # make separate tables for each channel
    for channel_name, channel_array in Plate.data:
        # the baseline data is at time index 0
        baseline_data = channel_array[0, ...]

        # make unique
        PrismColumns = []

        # go through each conjugate
        for conjugate in conjugates:
            plate_filter = Plate.Read_Plate.lookup(conjugate)
            filtered_data = baseline_data[plate_filter]

            # create a single YColumn for bar graph
            _column = createPrismColumn(
                filtered_data[:, np.newaxis],
                title="{:s}".format(conjugate),
                column_type="Y",
                decimal="4",
            )

            # append each conjugate to the column collection
            PrismColumns.append(_column)

        # create a table for the current channel
        baselineTable = assembleColumnPrismTable(
            PrismColumns, name="Baseline Data ({:s})".format(channel_name)
        )

        # add the table with conjugate data to Prism file
        prism_file.append(baselineTable)

    # finally, write out the Prism file
    prism_file.write(output_prism_file)
    print("{:s} has been successfully written.".format(output_prism_file))


def summarize_kinetics_data(
    Plate, output_prism_file, time_unit="minutes", dt=1.0, relative_change=True
):

    if relative_change:
        plate_data = Plate.relative_data
    else:
        plate_data = Plate.data

    # figure out how many unique combinations exist
    # figure out the number of unique experiments
    read_plates = np.unique(Plate.Read_Plate.substance)
    compounds = np.unique(Plate.Source_Plate_1.substance)
    Nconjugates = len(read_plates)
    Ncompounds = len(compounds)

    # pre-allocate table
    TableOfKinetics = []
    Ntime = Plate.Nbaseline + Plate.Npostinject

    # build array for kinetic data, first we need the time index
    time_array = np.arange(Ntime) * Plate.dt_postinject
    time_array.shape = (Ntime, 1)
    TimeColumn = createPrismColumn(
        time_array, title="Time ({:s})".format(time_unit), column_type="X"
    )

    for i, conjugate in enumerate(read_plates):
        for j, compound in enumerate(compounds):
            platefilter = Plate.Read_Plate.lookup(
                conjugate
            ) & Plate.Source_Plate_1.lookup(compound)

            # check if conjugate+compound pair exists, if so, proceed
            if platefilter.sum() > 0:
                # the the concentrations
                allconcs = Plate.Source_Plate_1.concentration[platefilter]
                unique_concs = np.unique(allconcs)
                unit = np.unique(Plate.Source_Plate_1.unit[platefilter])[0]

                # get some quantities here
                Nconcs = len(unique_concs)
                Nreplicate = int(len(allconcs) / len(unique_concs))

                # get concentration
                conc_data = Plate.Source_Plate_1.concentration[platefilter]

                # loop through every channel
                for ch_name, ch_data in plate_data:
                    filtered_data = ch_data[:, platefilter]
                    GroupColumns = []
                    # assemble a column grouped by unique concentrations
                    for conc in unique_concs:
                        conc_filter = conc_data == conc
                        wrkdata = filtered_data[:, conc_filter]
                        PrismColumn = createPrismColumn(
                            wrkdata,
                            title="{:.4f} {:s}".format(conc, unit),
                            column_type="Y",
                            decimal="4",
                        )
                        GroupColumns.append(PrismColumn)
                    # after all columns are collected for each concentrations
                    # put them into a single Table
                    Table1 = assembleXYPrismTable(
                        [TimeColumn],
                        GroupColumns,
                        Nreplicate,
                        name="{:s} + {:s} ({:s})".format(
                            conjugate, compound, ch_name
                        ),
                        ID="Table1",
                    )

            # append this conjugate + compound pair
            TableOfKinetics.append(Table1)

    # now then we make the PrismFile
    prism_outfile = PrismFile()

    for table in TableOfKinetics:
        prism_outfile.append(table)

    prism_outfile.write(output_prism_file)
    print("{:s} has been successfully written.".format(output_prism_file))


def summarize_last_read(Plate, output_prism_file, relative_change=True):
    # for regular expression
    ptn = lambda s: "^{:s}$".format(s)

    if relative_change:
        plate_data = Plate.relative_data
    else:
        plate_data = Plate.data

    # figure out how many unique combinations exist
    # figure out the number of unique experiments
    read_plate = np.unique(Plate.Read_Plate.substance)
    source_plate = np.unique(Plate.Source_Plate_1.substance)

    # create a new table for each conjugate
    prism_file = PrismFile()

    ConjugateTables = []

    for conjugate in read_plate:
        conjugate_filter = Plate.Read_Plate.lookup(conjugate)
        all_concs = np.unique(Plate.Source_Plate_1.concentration)

        # create an array for compound names
        compound_names = np.unique(source_plate)

        ConcentrationColumns = []

        # build our columns for each concentration
        for conc in all_concs:
            conc_filter = Plate.Source_Plate_1.concentration == conc

            # build our column of replicates
            replicate_list = []

            for compound in source_plate:
                compound_filter = Plate.Source_Plate_1.lookup(compound)
                _filter = conjugate_filter & conc_filter & compound_filter

                # assume every unit is the same, take the first one
                if _filter.sum() > 0:
                    _unit = np.unique(Plate.Source_Plate_1.unit[_filter])[0]
                else:
                    continue

                # get the last timepoint
                replicate_list.append(Plate.relative_data.pSHG[-1, _filter])

            # get the max replicate number for allocating Prism columns
            Nreplicate = max([len(sublist) for sublist in replicate_list])

            # if we find empty sublists, fill them with np.nan
            for i, sublist in enumerate(replicate_list):
                if len(sublist) == 0:
                    replicate_list[i] = np.repeat(np.nan, Nreplicate)

            replicate_array = np.array(replicate_list)

            GroupedColumns = createPrismColumn(
                replicate_array,
                title="{:.2f} {:s}".format(conc, _unit),
                column_type="Y",
                decimal="4",
            )
            ConcentrationColumns.append(GroupedColumns)

        # form a new table from the compounds grouped by its concentration
        ConjugateGroupedTable = assembleGroupedPrismTable(
            compound_names,
            ConcentrationColumns,
            Nreplicate,
            name="{:s} (P-SHG)".format(conjugate),
        )

        ConjugateTables.append(ConjugateGroupedTable)

    # combine all of the tables
    for table in ConjugateTables:
        prism_file.append(table)

    prism_file.write(output_prism_file)
    print("{:s} has been successfully written.".format(output_prism_file))


def assembleColumnPrismTable(column_list, name="Table Name", ID="Table0"):
    """ Put together the 'Column data table' for Prism

    The simplest data type commonly used to make bar graphs in Prism. The
    replicates are entered along rows. Each column corresponds to a different
    experimental sample/category.

    Args:
        column_list(list of PrismColumns): each column contains experimental
            replicates.
        name(str): the name for the data table as it will appear in Prism.
        ID(str): table number. This number is arbitrary and may be altered
            once you load the file in Prism.

    """

    # each column must not have replicates (e.g. only number of entries)
    Ncolumns = len(column_list)

    Table = Element(
        "Table",
        attrib={
            "EVFormat": "AsteriskAfterNumber",
            "ID": ID,
            "XFormat": "none",
            "TableType": "OneWay",
        },
    )

    Title = SubElement(Table, "Title")
    Title.text = name

    for column in column_list:
        Table.append(column)

    return Table


def assembleGroupedPrismTable(
    group_names, column_list, Nreplicates, name="Table Name", ID="Table0"
):
    """ Put together a Prism "Grouped Data Table"

    What is a Grouped table? it's commonly used to display bar graphs
    with some quantities (with replicates) that belong to some group (the
    labels provided along the rows, here it's the group_names argument) with
    different experimental settings (here is defined as the title for each
    PrismColumn). The PrismColumn is generated from the function
    :func:`PyBiodesy.Text2Prism.createPrismColumn`

    Args:
        group_names(list of str): the names the experimental group (row labels)
        column_list(list of PrismColumns): each column is a replicate
        Nreplicates(int): the number of replicates for all PrismColumn entry
        name(str): the name of data table as it will appear in Prism
        ID(str): table number. This number is arbitrary and may be altered
            once you load the file in Prism.

    """
    Ngroups = len(group_names)

    Table = Element(
        "Table",
        attrib={
            "EVFormat": "AsteriskAfterNumber",
            "ID": ID,
            "XFormat": "none",
            "YFormat": "replicates",
            "Replicates": str(Nreplicates),
            "TableType": "TwoWay",
        },
    )

    Title = SubElement(Table, "Title")
    Title.text = name

    # populate row titles for each group
    RowTitles = SubElement(Table, "RowTitlesColumn", attrib={"Width": "75"})
    Subcolumn = SubElement(RowTitles, "Subcolumn")

    for row_title in group_names:
        entry = SubElement(Subcolumn, "d")
        entry.text = row_title

    # then populate the YColumn
    for column in column_list:
        Table.append(column)

    return Table


def assembleXYPrismTable(
    x_column,
    y_columns,
    Nreplicates,
    row_titles="",
    name="Table Name",
    ID="Table0",
):
    """ Put together a Prism 'XY Data Table'

    Currently this function only supports X-axis data supplied as
    a list containing a signle vector (e.g. [nd.array]). The PrismColumn
    is generated from the function :func:`PyBiodesy.Text2Prism.createPrismColumn`

    Args:
        x_column(list): a list of X-axis data, a single NumPy array.
        y_columns(list of PrismColumns): a list of PrismColumns.
        Nreplicates(int): the number of replicate for all PrismColumn entry.
        row_titles(list of str): a list of strings with the same length as
            len(x_column[0]) for naming each element of the X-axis data.
        name(str): the name for the data table as it will appear in Prism
        ID(str): table number. This number is arbitrary and may be altered
            once you load the file in Prism.

    """
    Ngroups_y = len(y_columns)
    Ngroups_x = len(x_column)

    assert Ngroups_x == 1, "Only one X column is currently implemented"

    Table = Element(
        "Table",
        attrib={
            "EVFormat": "AsteriskAfterNumber",
            "ID": ID,
            "Replicates": str(int(Nreplicates)),
            "TableType": "XY",
            "XFormat": "numbers",
            "YFormat": "replicates",
        },
    )

    TableTitle = SubElement(Table, "Title")
    TableTitle.text = name

    if row_titles != "":
        RowTitles = SubElement(
            Table, "RowTitlesColumn", attrib={"Width": "50"}
        )
        rowSubcolumn = SubElement(Table, "Subcolumn")
        for row_title in row_titles:
            title_entry = SubElement(rowSubcolumn, "d")
            title_entry.text = str(row_title)

    for each_x in x_column:
        Table.append(each_x)

    for each_y in y_columns:
        Table.append(each_y)

    return Table


def createPrismColumn(
    data_array, title="Title", column_type="Y", width="80", decimal="2"
):
    """ create a Prism XML column tag populate with data_array

    Args:
        data_array(ndarray): 2-d numpy array, tabular data.
        title(str): the column label.
        column_type(str): either "X" or "Y". Default is "Y"
        width(int): width of column displayed in Prism. Default is 80
        decimal(int): number of digits after decimal point for display in Prism

    Returns:
        XML tag filled with input data

    """
    # the data needs to be shaped such that
    # data_array.shape[0] is Nentries
    # data_array.shape[1] is Nsubcolumns, can be thought of Nreplicates
    assert (
        data_array.ndim < 3
    ), "Input array must be 1- or 2-Dimensional for a Prism Column"

    if data_array.ndim == 1:
        # for 1-dimensional array (vector), it only has 1 sub-column
        # so 'broadcast' to new dimension
        data_array = data_array[:, np.newaxis]

    Nentries, Nsubcolumns = data_array.shape

    if column_type == "Y":
        column_tag = "YColumn"
    elif column_type == "X":
        column_tag = "XColumn"
    else:
        raise NotImplementedError(
            "Can't recognize column type {:s}.".format(column_type)
        )

    Column = Element(
        column_tag,
        attrib={
            "width": width,
            "Decimals": decimal,
            "subcolumns": str(Nsubcolumns),
        },
    )

    if title is not None:
        Title = SubElement(Column, "Title")
        Title.text = title

    # populate subcolumns
    for c in range(Nsubcolumns):
        Subcolumn = SubElement(Column, "Subcolumn")
        # populate each entry
        for i in range(Nentries):
            entry = SubElement(Subcolumn, "d")
            data_entry = data_array[i, c]
            # deal with missing data, insert blank
            if np.isnan(data_entry):
                entry.text = " "
            else:
                entry.text = str(data_entry)

    return Column


def summarize_kinetics_v2(Plate, relative_data=False):

    fileprefix = Plate.filename.with_suffix("").name
    relative_data_suffix = ("absolute_signal", "percent_change")[relative_data]
    filefolder = Plate.filename.with_suffix("") / relative_data_suffix / "ByTime"
    fileoutfmt = "{:s}_{:s}_{{:s}}.pzfx".format(
        fileprefix, relative_data_suffix
    )

    if not filefolder.exists():
        filefolder.mkdir(exist_ok=True, parents=True)

    fileprefix = Plate.filename.with_suffix("").name

    fileoutfmt = "{:s}_{{:s}}.pzfx".format(fileprefix)

    targetfilefmt = str(filefolder / fileoutfmt)

    # list to contain each table, for separate channels
    ch_dict_columns = dict((ch_name, []) for ch_name in Plate.channel_strlist)

    # the entire plate has the same acquisition time, so we can
    t_array = Plate.dt_postinject * np.arange(Plate.data.Nreads) / 60.0
    t_prism_column = createPrismColumn(
        t_array, column_type='X', title="Time, min"
    )

    max_Nrep = 0

    for data, (conjugate, compound) in Plate.unique_experiments(
        relative_data=relative_data
    ):

        for uniqueconj, exptdata in data.items():

            for ch_name, ch_data in exptdata.items():

                Nreads, Nrep, Nconc = ch_data['data'].shape

                if Nrep > max_Nrep:
                    max_Nrep = Nrep

                # build columns for each compound concentration
                # for each channel, accumulate columns into a list
                column_conc_list = []
                for c in range(Nconc):
                    column_ = createPrismColumn(
                        ch_data['data'][:, :, c],
                        title="{:.2f} {:s}".format(
                            ch_data['concentrations'][c], ch_data['unit']
                        ),
                    )
                    column_conc_list.append(column_)

                # assemble as a table
                ch_kinetic_table = assembleXYPrismTable(
                    [t_prism_column],
                    column_conc_list,
                    Nreplicates=Nrep,
                    name="{:s} with {:s}".format(uniqueconj, compound),
                )

                # store these list of columns
                ch_dict_columns[ch_name].append(ch_kinetic_table)

    # now loop through the tables in each channel
    for ch_name, ch_tables in ch_dict_columns.items():
        output_file = targetfilefmt.format(ch_name)
        prism_hdl = PrismFile()
        for table in ch_tables:
            prism_hdl.append(table)
        prism_hdl.write(output_file)


def summarize_CRC_v2(Plate, relative_data=False):
    """ Generates a Prism (pzfx) file from a 'Plate' grouped by Concentration

    Each channel will have its own pzfx file. Each Prism file will contain
    separate 'XY Table' with concentration on the X-axis, and SHG (or %SHG)
    on the Y-axis, with replicates grouped per time point.

    """

    relative_data_suffix = ("absolute_signal", "percent_change")[relative_data]
    filefolder = Plate.filename.with_suffix("") / relative_data_suffix / "ByConcentration"

    if not filefolder.exists():
        filefolder.mkdir(exist_ok=True, parents=True)

    fileprefix = Plate.filename.with_suffix("").name

    fileoutfmt = "{:s}_{:s}_{{:s}}.pzfx".format(
        fileprefix, relative_data_suffix
    )

    targetfilefmt = str(filefolder / fileoutfmt)
    # list to contain each table, for separate channels
    ch_dict_columns = dict((ch_name, []) for ch_name in Plate.channel_strlist)
    dt_min = Plate.dt_postinject / 60.0

    for data, (conjugate, compound) in Plate.unique_experiments(
        relative_data=relative_data
    ):

        for uniqueconj, exptdata in data.items():

            for ch_name, ch_data in exptdata.items():
                # for each channel, accumulate columns into a list
                column_time_list = []
                Nreads, Nrep, Nconc = ch_data['data'].shape

                # build columns for each compound concentration
                column_time_list = []
                for t in range(Nreads):
                    column_ = createPrismColumn(
                        ch_data['data'][t, :, :].T,
                        title="t={:.2f} min".format(t * dt_min),
                    )
                    column_time_list.append(column_)

                conc_prism_column = createPrismColumn(
                    ch_data['concentrations'],
                    column_type='X',
                    title="[{:s}] {:s}".format(compound, ch_data['unit']),
                )

                # assemble as a table
                ch_CRC_table = assembleXYPrismTable(
                    [conc_prism_column],
                    column_time_list,
                    Nreplicates=Nrep,
                    name="{:s} with {:s}".format(uniqueconj, compound),
                )

                # store these list of columns
                ch_dict_columns[ch_name].append(ch_CRC_table)

    # now loop through the tables in each channel
    for ch_name, ch_tables in ch_dict_columns.items():
        output_file = targetfilefmt.format(ch_name)
        prism_hdl = PrismFile()
        for table in ch_tables:
            prism_hdl.append(table)
        prism_hdl.write(output_file)


def summarize_kinetics_ratios(Plate, ratio_flag="TPFratio", relative_data=False):

    filefolder = Plate.filename.with_suffix("") / "ByTime"

    fileprefix = Plate.filename.with_suffix("").name
    relative_data_suffix = ("absolute_signal", "percent_change")[relative_data]

    fileoutfmt = "{:s}_{:s}_{{:s}}.pzfx".format(
        fileprefix, relative_data_suffix
    )

    if not filefolder.exists():
        filefolder.mkdir(exist_ok=True, parents=True)

    fileprefix = Plate.filename.with_suffix("").name

    fileoutfmt = "{:s}_{{:s}}.pzfx".format(fileprefix)

    targetfilefmt = str(filefolder / fileoutfmt)

    # list to contain each table, for separate channels
    # ch_dict_columns = dict((ch_name, []) for ch_name in Plate.channel_strlist)
    ch_dict_columns = dict([(ratio_flag, [])])

    # the entire plate has the same acquisition time, so we can
    t_array = Plate.dt_postinject * np.arange(Plate.data.Nreads) / 60.0
    t_prism_column = createPrismColumn(
        t_array, column_type='X', title="Time, min"
    )

    max_Nrep = 0

    for data, (conjugate, compound) in Plate.unique_experiments(
        relative_data=relative_data
    ):

        for uniqueconj, exptdata in data.items():

            if ratio_flag == "TPFratio":
                pTPFdata = exptdata['pTPF']['data']
                sTPFdata = exptdata['sTPF']['data']

                # prevent division by zero P/S
                ratio_data = pTPFdata / np.maximum(sTPFdata, 1e-08)

                # for each channel, accumulate columns into a list
                Nreads, Nrep, Nconc = ratio_data.shape
                ratio_conc = exptdata['pTPF']['concentrations']
                ratio_unit = exptdata['pTPF']['unit']
            else:
                raise NotImplementedError("SHG ratio not implemented")
                return None

            if Nrep > max_Nrep:
                max_Nrep = Nrep

            # build columns for each compound concentration
            column_conc_list = []
            for c in range(Nconc):
                column_ = createPrismColumn(
                    ratio_data[:, :, c],
                    title="{:.2f} {:s}".format(
                        ratio_conc[c], ratio_unit
                    ),
                )
                column_conc_list.append(column_)

            # assemble as a table
            ch_kinetic_table = assembleXYPrismTable(
                [t_prism_column],
                column_conc_list,
                Nreplicates=Nrep,
                name="{:s} with {:s}".format(uniqueconj, compound),
            )

            # store these list of columns
            ch_dict_columns[ratio_flag].append(ch_kinetic_table)

    # now loop through the tables in each channel
    for ch_name, ch_tables in ch_dict_columns.items():
        output_file = targetfilefmt.format(ch_name)
        prism_hdl = PrismFile()
        for table in ch_tables:
            prism_hdl.append(table)
        prism_hdl.write(output_file)


def summarize_by_conjugates(Plate):
    """ Create a Prism file with 'Grouped Tables' for each unique conjugate

    Each compound name will be placed as row titles with time as the YColumn
    group. (But how should we deal with multiple conjugate concentrations?)

    """

    # create a Prism file summary grouped by unique conjugate
    #

    pass
