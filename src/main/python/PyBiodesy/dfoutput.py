"""
===============
dfoutput module
===============

This module contains objects used for writing a pandas DataFrame object
into something other than csv, Excel, or whatever is available via the pandas
API. Currently, output to a GraphPad Prism file is supported.


"""

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import getpass
import datetime


def pprintxml(xmlobj):
    """ convenient function to print an XML tag """
    xmlstr = tostring(xmlobj)
    print(parseString(xmlstr).toprettyxml(indent="  "))


class PrismFile:
    """ a basic class to build up a Prism XML file

    the XML structure is built-up on self.root

    """

    def __init__(self):
        # root is the outermost XML tag enclosing all other tags
        self.root = Element(
            "GraphPadPrismFile", attrib={"PrismXMLVersion": "5.00"}
        )
        self.date_created = datetime.datetime.now().date().isoformat()
        self.time_created = (
            datetime.datetime.now()
            .replace(microsecond=0)
            .astimezone()
            .isoformat()
        )
        self.Ninfo = 0
        self.Ntables = 0

        tag_created = SubElement(self.root, "Created")
        tag_version = SubElement(
            tag_created,
            "OriginalVersion",
            attrib={
                "CreatedByProgram": "GraphPad Prism",
                "CreatedByVersion": "8.2.0.272",
                "Login": getpass.getuser(),
                "DateTime": self.time_created,
            },
        )

    def add_info(self, project_title=None, notes=None, **kwargs):
        """ adds an information metadata to the Prism file

        Any arbitrary notes can be added by passing a keyworded argument, or
        if spaces are needed for the metadata tag, pass a **kwargs_as_dict
        containing key-value pairs.

        """
        self.tag_info = Element(
            "Info",
            attrib={
                "ID": "Info{:d}".format(self.Ninfo)
            }
        )

        tag_title = SubElement(self.tag_info, "Title")
        tag_title.text = project_title if project_title is not None else ""

        tag_notes = SubElement(self.tag_info, "Notes")
        tag_notes.text = notes if notes is not None else ""

        tag_const = SubElement(self.tag_info, "Constant")
        tag_name = SubElement(tag_const, "Name")
        tag_name.text = "Experiment Date"
        tag_value = SubElement(tag_const, "Value")
        tag_value.text = self.date_created

        if kwargs:
            for key, value in kwargs.items():
                tag_const = SubElement(self.tag_info, "Constant")
                tag_name = SubElement(tag_const, "Name")
                tag_name.text = str(key)
                tag_value = SubElement(tag_const, "Value")
                tag_value.text = str(value)

        self.root.insert(1, self.tag_info)

        self.Ninfo += 1

    def add_ColumnTable(self, df, table_title=None, use_index=False):
        """ appends a Prism Column Table to the current file

        Input dataframe must have one levels of column indices. You can also
        assign labels to each row by passing use_index=True.

        """
        deftitle = "Data {:d}".format(self.Ntables)
        tag_table = SubElement(
            self.root,
            "Table",
            attrib={
                "ID": "Table{:d}".format(self.Ntables),
                "XFormat": "none",
                "TableType": "OneWay",
                "EVFormat": "AsteriskAfterNumber",
            },
        )
        tag_title = SubElement(tag_table, "Title")
        tag_title.text = table_title if table_title is not None else deftitle

        # row titles (if desired)
        tag_rowtitlescolumn = SubElement(
            tag_table,
            "RowTitlesColumn",
            attrib={"Width": "81" if use_index else "1"},
        )
        tag_subcolumn = SubElement(tag_rowtitlescolumn, "Subcolumn")

        if use_index:
            for rowtitle in df.index.tolist():
                entry = SubElement(tag_subcolumn, "d")
                entry.text = (
                    rowtitle if isinstance(rowtitle, str) else str(rowtitle)
                )

        # get header names from dataframe
        NColumnLevels = df.columns.nlevels

        if NColumnLevels != 1:
            raise ValueError(
                (
                    "Input dataframe must only have 1 level of columns, "
                    "but it has {:d} levels"
                ).format(NColumnLevels)
            )

        # Get column header names
        YColumns = df.columns.tolist()

        for column in YColumns:
            ycolumn = SubElement(
                tag_table,
                "YColumn",
                attrib={"Decimals": "3", "Subcolumns": "1", "Width": "81"},
            )
            columnhdr = SubElement(ycolumn, "Title")
            columnhdr.text = column
            subarray = df[column].values

            if subarray.ndim != 1:
                raise ValueError("You can only have a single column per header")

            # one subcolumn per YColumn
            subcolumn = SubElement(ycolumn, "Subcolumn")

            for row in subarray:
                entry = SubElement(subcolumn, "d")
                entry.text = str(row) if row is not None else ""

        self.Ntables += 1

    def add_GroupedTable(self, df, table_title=None, use_index=False):
        """ appends a Prism Grouped Table to the current file

        Input dataframe must have two levels of column indices, with the
        second level being the replicates.

        """
        # default table title
        deftitle = "Data {:d}".format(self.Ntables)

        # get header names from dataframe
        NColumnLevels = df.columns.nlevels
        # second level is for replicates
        Nreplicates = len(set(df.columns.codes[1].tolist()))

        if NColumnLevels != 2:
            raise ValueError(
                (
                    "Input dataframe must only have 2 level of columns, "
                    "but it has {:d} levels"
                ).format(NColumnLevels)
            )

        tag_table = SubElement(
            self.root,
            "Table",
            attrib={
                "ID": "Table{:d}".format(self.Ntables),
                "Replicates": str(Nreplicates),
                "XFormat": "none",
                "YFormat": "replicates",
                "TableType": "TwoWay",
                "EVFormat": "AsteriskAfterNumber",
            },
        )
        tag_title = SubElement(tag_table, "Title")
        tag_title.text = table_title if table_title is not None else deftitle

        # row titles (if desired)
        tag_rowtitlescolumn = SubElement(
            tag_table,
            "RowTitlesColumn",
            attrib={"Width": "81" if use_index else "1"},
        )
        tag_subcolumn = SubElement(tag_rowtitlescolumn, "Subcolumn")

        if use_index:
            for rowtitle in df.index.tolist():
                entry = SubElement(tag_subcolumn, "d")
                entry.text = (
                    rowtitle if isinstance(rowtitle, str) else str(rowtitle)
                )

        # Get column header names, from top level, index-0
        YColumns = df.columns.levels[0].tolist()

        for column in YColumns:
            # get a subset of the current column
            subarray = df[column].values
            ycolumn = SubElement(
                tag_table,
                "YColumn",
                attrib=(
                    {
                        "Decimals": "3",
                        "Subcolumns": str(Nreplicates),
                        "Width": str(81 * Nreplicates),
                    }
                ),
            )
            columnhdr = SubElement(ycolumn, "Title")
            columnhdr.text = str(column)
            # to loop through a Numpy array, first transpose it
            # because the iterable returns each row
            for column in subarray.T:
                # each column is a replicate
                subcolumn = SubElement(ycolumn, "Subcolumn")
                for row in column:
                    entry = SubElement(subcolumn, "d")
                    entry.text = str(row) if row is not None else ""

        self.Ntables += 1

    def add_XYColumnTable(self, xseries, df, table_title=None, use_index=False):
        """ appends a Prism XY Table to the current file

        xseries is a pandas series for the data along x-axis. The input
        dataframe, df, must may have either one or two levels.

        """
        # default table title
        deftitle = "Data {:d}".format(self.Ntables)

        # get header names from dataframe
        NColumnLevels = df.columns.nlevels
        if NColumnLevels > 1:
            # second level is for replicates
            Nreplicates = len(set(df.columns.codes[1].tolist()))
        elif NColumnLevels == 1:
            Nreplicates = 1
        elif NColumnLevels > 2:
            raise ValueError(
                (
                    "Input dataframe must only have 1 or 2 level of columns, "
                    "but it has {:d} levels"
                ).format(NColumnLevels)
            )

        tag_table = SubElement(
            self.root,
            "Table",
            attrib={
                "ID": "Table{:d}".format(self.Ntables),
                "Replicates": str(Nreplicates),
                "XFormat": "numbers",
                "YFormat": "replicates",
                "TableType": "XY",
                "EVFormat": "AsteriskAfterNumber",
            },
        )
        tag_title = SubElement(tag_table, "Title")
        tag_title.text = table_title if table_title is not None else deftitle

        # row titles (if desired)
        tag_rowtitlescolumn = SubElement(
            tag_table,
            "RowTitlesColumn",
            attrib={"Width": "81" if use_index else "1"},
        )
        tag_subcolumn = SubElement(tag_rowtitlescolumn, "Subcolumn")

        if use_index:
            for rowtitle in df.index.tolist():
                entry = SubElement(tag_subcolumn, "d")
                entry.text = (
                    rowtitle if isinstance(rowtitle, str) else str(rowtitle)
                )

        # add X-data from xseries
        xcolumn = SubElement(
            tag_table,
            "XColumn",
            attrib=(
                {
                    "Decimals": "1",
                    "Subcolumns": "1",
                    "Width": "74",
                }
            )
        )
        # X-axis title
        xtitle = SubElement(xcolumn, "Title")
        xtitle.text = xseries.name
        subcolumn = SubElement(xcolumn, "Subcolumn")
        # add data for X-axis
        for row in xseries.values:
            entry = SubElement(subcolumn, "d")
            entry.text = str(row)

        # Get column header names, from top level, index-0
        YColumns = df.columns.levels[0].tolist()
        # then add Y-data
        for column in YColumns:
            # get a subset of the current column
            subarray = df[column].values
            ycolumn = SubElement(
                tag_table,
                "YColumn",
                attrib=(
                    {
                        "Decimals": "3",
                        "Subcolumns": str(Nreplicates),
                        "Width": str(81 * Nreplicates),
                    }
                ),
            )
            columnhdr = SubElement(ycolumn, "Title")
            columnhdr.text = column
            # to loop through a Numpy array, first transpose it
            # because the iterable returns each row
            for column in subarray.T:
                # each column is a replicate
                subcolumn = SubElement(ycolumn, "Subcolumn")
                for row in column:
                    entry = SubElement(subcolumn, "d")
                    entry.text = str(row) if row is not None else ""

        self.Ntables += 1

    def write(self, output_file):
        xmlutf8 = tostring(self.root, encoding="utf-8")
        prettyxml = parseString(xmlutf8).toprettyxml(indent="    ")
        with open(output_file, "w") as fh:
            fh.write(prettyxml)
