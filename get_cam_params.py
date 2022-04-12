from xml.etree import ElementTree as et
import argparse
import sys
import os
import subprocess
from colorama import Fore, Style


class XMLMerger:
    def __init__(self, pIntCamParams, pExtMarkersParams):
        self.roots = [et.parse(f).getroot() for f in (pIntCamParams, pExtMarkersParams)]

    def combineFiles(self):
        for r in self.roots[1:]:
            self.combineElement(self.roots[0], r)
        return et.tostring(self.roots[0])

    def combineElement(self, pFirstFile, pSecondFile):
        mapping = {el.tag: el for el in pFirstFile}
        for el in pSecondFile:
            if len(el) == 0:
                try:
                    mapping[el.tag].text = el.text
                except KeyError:
                    mapping[el.tag] = el
                    pFirstFile.append(el)
            else:
                try:
                    self.combineElement(mapping[el.tag], el)
                except KeyError:
                    mapping[el.tag] = el
                    pFirstFile.append(el)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f1", "--xmlC", required=False,
                    help="path to the internal cam params XML file")
    ap.add_argument("-f2", "--xmlM", required=False,
                    help="path to the markers params XML file")
    ap.add_argument("-f3", "--image", required=False,
                    help="path to the input image which will be analyzed")
    args = vars(ap.parse_args())

    xmlC = None
    xmlM = None
    image = None

    print()

    if (len(sys.argv) > 3):
        xmlC = args["xmlC"]
        xmlM = args["xmlM"]
        image = args["image"]
    else:
        if xmlC is None:
            xmlC = str(input(Fore.MAGENTA + "Enter path to the internal cam params XML file: " + Style.RESET_ALL))
        if xmlM is None:
            xmlM = str(input(Fore.MAGENTA + "Enter path to the markers params XML file: " + Style.RESET_ALL))
        if image is None:
            image = str(
                input(Fore.MAGENTA + "Enter path to the input image which will be analyzed: " + Style.RESET_ALL))

    xmlIntCamPar = xmlC
    xmlExtCamPar = "myCamExtParams.xml"
    xmlCamPar = "myCamParams.xml"

    launchHpm = "./hpm " + xmlC + " " + xmlM + " " + image + " --camera-position-calibration"

    try:
        os.remove(xmlExtCamPar)
    except:
        print()

    subprocess.call(launchHpm, shell=True)

    try:
        merger = XMLMerger(xmlIntCamPar, xmlExtCamPar)
        result = merger.combineFiles()

        file = open(xmlCamPar, "w")
        file.write("<?xml version='1.0'?>\n")
        file.write(result.decode("utf-8"))

        file.close()

        print()
        print(Fore.GREEN + "XML files merged succesfully!" + Style.RESET_ALL)
        print()
        print(
            Fore.YELLOW + "All camera parameters are located in the " + Style.RESET_ALL + xmlCamPar + Fore.YELLOW + " XML file!" + Style.RESET_ALL)
        print()
    except:
        print()
        print(Fore.RED + "Could not create XML file!" + Style.RESET_ALL)

if __name__ == '__main__':
    main()

