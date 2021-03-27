#!/usr/bin/env python

import os, sys, re, string, fnmatch
allmodules = ["core", "flann", "imgproc", "ml", "highgui", "video", "features2d", "calib3d", "objdetect", "legacy", "contrib", "gpu", "androidcamera", "java", "python", "stitching", "ts", "photo", "nonfree", "videostab", "ocl", "superres"]
verbose = False
show_warnings = True
show_errors = True
show_critical_errors = True

params_blacklist = {
    "fromarray" : ("object", "allowND"), # python only function
    "reprojectImageTo3D" : ("ddepth"),   # python only argument
    "composeRT" : ("d*d*"),              # wildchards in parameter names are not supported by this parser
    "CvSVM::train_auto" : ("\\*Grid"),   # wildchards in parameter names are not supported by this parser
    "error" : "args", # parameter of supporting macro
    "getConvertElem" : ("from", "cn", "to", "beta", "alpha"), # arguments of returned functions
    "gpu::swapChannels" : ("dstOrder") # parameter is not parsed correctly by the hdr_parser
}

ERROR_001_SECTIONFAILURE      = 1
WARNING_002_HDRWHITESPACE     = 2
ERROR_003_PARENTHESES         = 3
WARNING_004_TABS              = 4
ERROR_005_REDEFENITIONPARAM   = 5
ERROR_006_REDEFENITIONFUNC    = 6
WARNING_007_UNDOCUMENTEDPARAM = 7
WARNING_008_MISSINGPARAM      = 8
WARNING_009_HDRMISMATCH       = 9
ERROR_010_NOMODULE            = 10
ERROR_011_EOLEXPECTED         = 11

params_mapping = {
    "composeRT" : {
        "dr3dr1" : "d*d*",
        "dr3dr2" : "d*d*",
        "dr3dt1" : "d*d*",
        "dr3dt2" : "d*d*",
        "dt3dr1" : "d*d*",
        "dt3dr2" : "d*d*",
        "dt3dt1" : "d*d*",
        "dt3dt2" : "d*d*"
        },
    "CvSVM::train_auto" : {
        "coeffGrid" : "\\*Grid",
        "degreeGrid" : "\\*Grid",
        "gammaGrid" : "\\*Grid",
        "nuGrid" : "\\*Grid",
        "pGrid" : "\\*Grid"
    }
}

known_text_sections_names = ["Appendix", "Results", "Prerequisites", "Introduction", "Description"]

class DeclarationParser(object):
    def __init__(self, line=None):
        if line is None:
            self.fdecl = ""
            self.lang = ""
            self.balance = 0
            return
        self.lang = self.getLang(line)
        assert self.lang is not None
        self.fdecl = line[line.find("::")+2:].strip()
        self.balance = self.fdecl.count("(") - self.fdecl.count(")")
        assert self.balance >= 0

    def append(self, line):
        self.fdecl += line
        self.balance = self.fdecl.count("(") - self.fdecl.count(")")

    def isready(self):
        return self.balance == 0

    @classmethod
    def getLang(cls, line):
        if line.startswith(".. ocv:function::"):
            return "C++"
        if line.startswith(".. ocv:cfunction::"):
            return "C"
        if line.startswith(".. ocv:pyfunction::"):
            return "Python2"
        if line.startswith(".. ocv:pyoldfunction::"):
            return "Python1"
        if line.startswith(".. ocv:jfunction::"):
            return "Java"
        return None

    def hasDeclaration(self, line):
        return self.getLang(line) is not None

class ParamParser(object):
    def __init__(self, line=None):
        if line is None:
            self.prefix = ""
            self.name = ""
            self.comment = ""
            self.active = False
            return
        offset = line.find(":param")
        assert offset > 0
        self.prefix = line[:offset]
        assert self.prefix == " "*len(self.prefix), ":param definition should be prefixed with spaces"
        line = line[offset + 6:].lstrip()
        name_end = line.find(":")
        assert name_end > 0
        self.name = line[:name_end]
        self.comment = line[name_end+1:].lstrip()
        self.active = True

    def append(self, line):
        assert self.active
        if (self.hasDeclaration(line)):
            self.active = False
        elif line.startswith(self.prefix) or not line:
            self.comment += "\n" + line.lstrip()
        else:
            self.active = False

    @classmethod
    def hasDeclaration(cls, line):
        return line.lstrip().startswith(":param")

class RstParser(object):
    def __init__(self, cpp_parser):
        self.cpp_parser = cpp_parser
        self.definitions = {}
        self.sections_parsed = 0
        self.sections_total = 0
        self.sections_skipped = 0

    def parse(self, module_name, module_path=None):
        if module_path is None:
            module_path = "../" + module_name

        doclist = []
        for root, dirs, files in os.walk(os.path.join(module_path,"doc")):
            for filename in fnmatch.filter(files, "*.rst"):
                doclist.append(os.path.join(root, filename))

        for doc in doclist:
            self.parse_rst_file(module_name, doc)

    def parse_section_safe(self, module_name, section_name, file_name, lineno, lines):
        try:
            self.parse_section(module_name, section_name, file_name, lineno, lines)
        except AssertionError, args:
            if show_errors:
                print >> sys.stderr, "RST parser error E%03d: assertion in \"%s\" at %s:%s" % (ERROR_001_SECTIONFAILURE, section_name, file_name, lineno)
                print >> sys.stderr, "    Details: %s" % args

    def parse_section(self, module_name, section_name, file_name, lineno, lines):
        self.sections_total += 1
        # skip sections having whitespace in name
        #if section_name.find(" ") >= 0 and section_name.find("::operator") < 0:
        if (section_name.find(" ") >= 0 and not bool(re.match(r"(\w+::)*operator\s*(\w+|>>|<<|\(\)|->|\+\+|--|=|==|\+=|-=)", section_name)) ) or section_name.endswith(":"):
            if show_errors:
                print >> sys.stderr, "RST parser warning W%03d:  SKIPPED: \"%s\" File: %s:%s" % (WARNING_002_HDRWHITESPACE, section_name, file_name, lineno)
            self.sections_skipped += 1
            return

        func = {}
        func["name"] = section_name
        func["file"] = file_name
        func["line"] = lineno
        func["module"] = module_name

        # parse section name
        section_name = self.parse_namespace(func, section_name)
        class_separator_idx = section_name.find("::")
        if class_separator_idx > 0:
            func["class"] = section_name[:class_separator_idx]
            func["method"] = section_name[class_separator_idx+2:]
        else:
            func["method"] = section_name

        capturing_seealso = False
        skip_code_lines = False
        expected_brief = True
        was_code_line = False
        fdecl = DeclarationParser()
        pdecl = ParamParser()
        ll = None

        for l in lines:
            # read tail of function/method declaration if needed
            if not fdecl.isready():
                fdecl.append(ll)
                if fdecl.isready():
                    self.add_new_fdecl(func, fdecl)
                continue

            # continue capture seealso
            if capturing_seealso:
                if not l or l.startswith(" "):
                    seealso = func.get("seealso", [])
                    seealso.extend(l.split(","))
                    func["seealso"] = seealso
                    continue
                else:
                    capturing_seealso = False

            ll = l.strip()
            if ll == "..":
                expected_brief = False
                skip_code_lines = False
                continue

            # skip lines if line-skipping mode is activated
            if skip_code_lines:
                if not l:
                    continue
                if not l.startswith(" "):
                    skip_code_lines = False

            if ll.startswith(".. code-block::") or ll.startswith(".. image::"):
                skip_code_lines = True

                continue

            # todo: parse structure members; skip them for now
            if ll.startswith(".. ocv:member::"):
        #print ll
                skip_code_lines = True
                continue

            #ignore references (todo: collect them)
            if l.startswith(".. ["):
                continue

            if ll.startswith(".. "):
                expected_brief = False
            elif ll.endswith("::"):
                # turn on line-skipping mode for code fragments
                #print ll
                skip_code_lines = True
                ll = ll[:len(ll)-2]

            # continue param parsing (process params after processing .. at the beginning of the line and :: at the end)
            if pdecl.active:
                pdecl.append(l)
                if pdecl.active:
                    continue
                else:
                    self.add_new_pdecl(func, pdecl)
                    # do not continue - current line can contain next parameter definition

            # parse ".. seealso::" blocks
            if ll.startswith(".. seealso::"):
                if ll.endswith(".. seealso::"):
                    capturing_seealso = True
                else:
                    seealso = func.get("seealso", [])
                    seealso.extend(ll[ll.find("::")+2:].split(","))
                    func["seealso"] = seealso
                continue

            # skip ".. index::"
            if ll.startswith(".. index::"):
                continue

            # parse class & struct definitions
            if ll.startswith(".. ocv:class::"):
                func["class"] = ll[ll.find("::")+2:].strip()
                if "method" in func:
                    del func["method"]
                func["isclass"] = True
                expected_brief = True
                continue

            if ll.startswith(".. ocv:struct::"):
                func["class"] = ll[ll.find("::")+2:].strip()
                if "method" in func:
                    del func["method"]
                func["isstruct"] = True
                expected_brief = True
                continue

            # parse function/method definitions
            if fdecl.hasDeclaration(ll):
                fdecl = DeclarationParser(ll)
                if fdecl.isready():
                    self.add_new_fdecl(func, fdecl)
                continue

            # parse parameters
            if pdecl.hasDeclaration(l):
                pdecl = ParamParser(l)
                continue

            # record brief description
            if expected_brief:
                func["brief"] = func.get("brief", "") + "\n" + ll
                if skip_code_lines:
                    expected_brief = False # force end brief if code block begins
                continue

            # record other lines as long description
            if (skip_code_lines):
                ll = ll.replace("/*", "/ *")
                ll = ll.replace("*/", "* /")
                if (was_code_line):
                    func["long"] = func.get("long", "") + "\n" + ll + "\n"
                else:
                    was_code_line = True
                    func["long"] = func.get("long", "") + ll +"\n<code>\n\n // C++ code:\n\n"
            else:
                if (was_code_line):
                    func["long"] = func.get("long", "") + "\n" + ll + "\n</code>\n"
                    was_code_line = False
                else:
                    func["long"] = func.get("long", "") + "\n" + ll
        # endfor l in lines

        if fdecl.balance != 0:
            if show_critical_errors:
                print >> sys.stderr, "RST parser error E%03d: invalid parentheses balance in \"%s\" at %s:%s" % (ERROR_003_PARENTHESES, section_name, file_name, lineno)
            return

        # save last parameter if needed
        if pdecl.active:
            self.add_new_pdecl(func, pdecl)

        # add definition to list
        func = self.normalize(func)
        if self.validate(func):
            self.definitions[func["name"]] = func
            self.sections_parsed += 1
            if verbose:
                self.print_info(func)
        elif func:
            if func["name"] in known_text_sections_names:
                if show_errors:
                    print >> sys.stderr, "RST parser warning W%03d:  SKIPPED: \"%s\" File: %s:%s" % (WARNING_002_HDRWHITESPACE, section_name, file_name, lineno)
                self.sections_skipped += 1
            elif show_errors:
                self.print_info(func, True, sys.stderr)

    def parse_rst_file(self, module_name, doc):
        doc = os.path.abspath(doc)
        lineno = 0
        whitespace_warnings = 0
        max_whitespace_warnings = 10

        lines = []
        flineno = 0
        fname = ""
        prev_line = None

        df = open(doc, "rt")
        for l in df.readlines():
            lineno += 1
            # handle tabs
            if l.find("\t") >= 0:
                whitespace_warnings += 1
                if whitespace_warnings <= max_whitespace_warnings and show_warnings:
                    print >> sys.stderr, "RST parser warning W%03d: tab symbol instead of space is used at %s:%s" % (WARNING_004_TABS, doc, lineno)
                l = l.replace("\t", "    ")

            # handle first line
            if prev_line == None:
                prev_line = l.rstrip()
                continue

            ll = l.rstrip()
            if len(prev_line) > 0 and len(ll) >= len(prev_line) and (ll == "-" * len(ll) or ll == "+" * len(ll) or ll == "=" * len(ll)):
                # new function candidate
                if len(lines) > 1:
                    self.parse_section_safe(module_name, fname, doc, flineno, lines[:len(lines)-1])
                lines = []
                flineno = lineno-1
                fname = prev_line.strip()
            elif flineno > 0:
                lines.append(ll)
            prev_line = ll
        df.close()

        # don't forget about the last function section in file!!!
        if len(lines) > 1:
            self.parse_section_safe(module_name, fname, doc, flineno, lines)

    @classmethod
    def parse_namespace(cls, func, section_name):
        known_namespaces = ["cv", "gpu", "flann", "superres"]
        l = section_name.strip()
        for namespace in known_namespaces:
            if l.startswith(namespace + "::"):
                func["namespace"] = namespace
                return l[len(namespace)+2:]
        return section_name

    def add_new_fdecl(self, func, decl):
        if decl.fdecl.endswith(";"):
            print >> sys.stderr, "RST parser error E%03d: unexpected semicolon at the end of declaration in \"%s\" at %s:%s" \
                        % (ERROR_011_EOLEXPECTED, func["name"], func["file"], func["line"])
        decls =  func.get("decls", [])
        if (decl.lang == "C++" or decl.lang == "C"):
            rst_decl = self.cpp_parser.parse_func_decl_no_wrap(decl.fdecl)
            decls.append( [decl.lang, decl.fdecl, rst_decl] )
        else:
            decls.append( [decl.lang, decl.fdecl] )
        func["decls"] = decls

    @classmethod
    def add_new_pdecl(cls, func, decl):
        params =  func.get("params", {})
        if decl.name in params:
            if show_errors:
                #check black_list
                if decl.name not in params_blacklist.get(func["name"], []):
                    print >> sys.stderr, "RST parser error E%03d: redefinition of parameter \"%s\" in \"%s\" at %s:%s" \
                        % (ERROR_005_REDEFENITIONPARAM, decl.name, func["name"], func["file"], func["line"])
        else:
            params[decl.name] = decl.comment
            func["params"] = params

    def print_info(self, func, skipped=False, out = sys.stdout):
        print >> out
        if skipped:
            print >> out, "SKIPPED DEFINITION:"
        print >> out, "name:      %s" % (func.get("name","~empty~"))
        print >> out, "file:      %s:%s" % (func.get("file","~empty~"), func.get("line","~empty~"))
        print >> out, "is class:  %s" % func.get("isclass", False)
        print >> out, "is struct: %s" % func.get("isstruct", False)
        print >> out, "module:    %s" % func.get("module","~unknown~")
        print >> out, "namespace: %s" % func.get("namespace", "~empty~")
        print >> out, "class:     %s" % (func.get("class","~empty~"))
        print >> out, "method:    %s" % (func.get("method","~empty~"))
        print >> out, "brief:     %s" % (func.get("brief","~empty~"))
        if "decls" in func:
            print >> out, "declarations:"
            for d in func["decls"]:
                print >> out, "     %7s: %s" % (d[0], re.sub(r"[ ]+", " ", d[1]))
        if "seealso" in func:
            print >> out, "seealso:  ", func["seealso"]
        if "params" in func:
            print >> out, "parameters:"
            for name, comment in func["params"].items():
                print >> out, "%23s:   %s" % (name, comment)
        print >> out, "long:      %s" % (func.get("long","~empty~"))
        print >> out

    def validate(self, func):
        if func.get("decls", None) is None:
            if not func.get("isclass", False) and not func.get("isstruct", False):
                return False
        if func["name"] in self.definitions:
            if show_errors:
                print >> sys.stderr, "RST parser error E%03d: \"%s\" from: %s:%s is already documented at %s:%s" \
                    % (ERROR_006_REDEFENITIONFUNC, func["name"], func["file"], func["line"], self.definitions[func["name"]]["file"], self.definitions[func["name"]]["line"])
            return False
        return self.validateParams(func)

    def validateParams(self, func):
        documentedParams = func.get("params", {}).keys()
        params = []

        for decl in func.get("decls", []):
            if len(decl) > 2:
                args = decl[2][3] # decl[2] -> [ funcname, return_ctype, [modifiers], [args] ]
                for arg in args:
                    # arg -> [ ctype, name, def val, [mod], argno ]
                    if arg[0] != "...":
                        params.append(arg[1])
        params = list(set(params))#unique

        # 1. all params are documented
        for p in params:
            if p not in documentedParams and show_warnings:
                print >> sys.stderr, "RST parser warning W%03d: parameter \"%s\" of \"%s\" is undocumented. %s:%s" % (WARNING_007_UNDOCUMENTEDPARAM, p, func["name"], func["file"], func["line"])

        # 2. only real params are documented
        for p in documentedParams:
            if p not in params and show_warnings:
                if p not in params_blacklist.get(func["name"], []):
                    print >> sys.stderr, "RST parser warning W%03d: unexisting parameter \"%s\" of \"%s\" is documented at %s:%s" % (WARNING_008_MISSINGPARAM, p, func["name"], func["file"], func["line"])
        return True

    def normalize(self, func):
        if not func:
            return func
        fnname = func["name"]
        fnname = self.normalizeText(fnname)
        fnname = re.sub(r'_\?D$', "_nD", fnname)  # tailing _?D can be mapped to _nD
        fnname = re.sub(r'\?D$', "ND", fnname)  # tailing ?D can be mapped to ND
        fnname = re.sub(r'\(s\)$', "s", fnname) # tailing (s) can be mapped to s
        func["name"] = fnname
        if "method" in func:
            func["method"] = self.normalizeText(func["method"])
        if "class" in func:
            func["class"] = self.normalizeText(func["class"])
        if "brief" in func:
            func["brief"] = self.normalizeText(func.get("brief", None))
            if not func["brief"]:
                del func["brief"]
        if "long" in func:
            func["long"] = self.normalizeText(func.get("long", None))
            if not func["long"]:
                del func["long"]
        if "decls" in func:
            func["decls"].sort()
        if "params" in func:
            params = {}
            for name, comment in func["params"].items():
                cmt = self.normalizeText(comment)
                if cmt:
                    params[name] = cmt
            # expand some wellknown params
            pmap = params_mapping.get(fnname)
            if pmap:
                for name, alias in pmap.items():
                    params[name] = params[alias]
            func["params"] = params
        if "seealso" in func:
            seealso = []
            for see in func["seealso"]:
                item = self.normalizeText(see.rstrip(".")).strip("\"")
                if item and (item.find(" ") < 0 or item.find("::operator") > 0):
                    seealso.append(item)
            func["seealso"] = list(set(seealso))
            if not func["seealso"]:
                del func["seealso"]

        # special case for old C functions - section name should omit "cv" prefix
        if not func.get("isclass", False) and not func.get("isstruct", False):
            self.fixOldCFunctionName(func)
        return func

    def fixOldCFunctionName(self, func):
        if not "decls" in func:
            return
        fname = None
        for decl in func["decls"]:
            if decl[0] != "C" and decl[0] != "Python1":
                return
            if decl[0] == "C":
                fname = decl[2][0]
        if fname is None:
            return

        fname = fname.replace(".", "::")
        if fname.startswith("cv::cv"):
            if fname[6:] == func.get("name", "").replace("*", "_n"):
                func["name"] = fname[4:]
                func["method"] = fname[4:]
            elif show_warnings:
                print >> sys.stderr, "RST parser warning W%03d:  \"%s\" - section name is \"%s\" instead of \"%s\" at %s:%s" % (WARNING_009_HDRMISMATCH, fname, func["name"], fname[6:], func["file"], func["line"])
                #self.print_info(func)

    def normalizeText(self, s):
        if s is None:
            return s

        s = re.sub(r"\.\. math::[ \r]*\n+((.|\n)*?)(\n[ \r]*\n|$)", mathReplace2, s)
        s = re.sub(r":math:`([^`]+?)`", mathReplace, s)
        s = re.sub(r" *:sup:", "^", s)

        s = s.replace(":ocv:class:", "")
        s = s.replace(":ocv:struct:", "")
        s = s.replace(":ocv:func:", "")
        s = s.replace(":ocv:cfunc:","")
        s = s.replace(":c:type:", "")
        s = s.replace(":c:func:", "")
        s = s.replace(":ref:", "")
        s = s.replace(":math:", "")
        s = s.replace(":func:", "")

        s = s.replace("]_", "]")
        s = s.replace(".. note::", "Note:")
        s = s.replace(".. table::", "")
        s = s.replace(".. ocv:function::", "")
        s = s.replace(".. ocv:cfunction::", "")

        # remove ".. identifier:" lines
        s = re.sub(r"(^|\n)\.\. [a-zA-Z_0-9]+(::[a-zA-Z_0-9]+)?:(\n|$)", "\n ", s)
        # unwrap urls
        s = re.sub(r"`([^`<]+ )<(https?://[^>]+)>`_", "\\1(\\2)", s)
        # remove tailing ::
        s = re.sub(r"::(\n|$)", "\\1", s)

        # normalize line endings
        s = re.sub(r"\r\n", "\n", s)
        # remove extra line breaks before/after _ or ,
        s = re.sub(r"\n[ ]*([_,])\n", r"\1 ", s)
        # remove extra line breaks after `
        #s = re.sub(r"`\n", "` ", s)
        # remove extra space after ( and before .,)
        s = re.sub(r"\([\n ]+", "(", s)
        s = re.sub(r"[\n ]+(\.|,|\))", "\\1", s)
        # remove extra line breaks after ".. note::"
        s = re.sub(r"\.\. note::\n+", ".. note:: ", s)
        # remove extra line breaks before *
        s = re.sub(r"\n+\*", "\n*", s)
        # remove extra line breaks after *
        s = re.sub(r"\n\*\n+", "\n* ", s)
        # remove extra line breaks before #.
        s = re.sub(r"\n+#\.", "\n#.", s)
        # remove extra line breaks after #.
        s = re.sub(r"\n#\.\n+", "\n#. ", s)
        # remove extra line breaks before `
        #s = re.sub(r"\n[ ]*`", " `", s)
        # remove trailing whitespaces
        s = re.sub(r"[ ]+$", "", s)
        # remove .. for references
        #s = re.sub(r"\.\. \[", "[", s)
        # unescape
        s = re.sub(r"\\(.)", "\\1", s)

        # remove whitespace before .
        s = re.sub(r"[ ]+\.", ".", s)
        # remove tailing whitespace
        s = re.sub(r" +(\n|$)", "\\1", s)
        # remove leading whitespace
        s = re.sub(r"(^|\n) +", "\\1", s)
        # compress line breaks
        s = re.sub(r"\n\n+", "\n\n", s)
        # remove other newlines
        s = re.sub(r"([^.\n\\=])\n([^*#\n]|\*[^ ])", "\\1 \\2", s)
        # compress whitespace
        s = re.sub(r" +", " ", s)

        # restore math
        s = re.sub(r" *<BR> *", "\n", s)

        # remove extra space before .
        s = re.sub(r"[\n ]+\.", ".", s)

        s = s.replace("**", "")
        s = re.sub(r"``([^\n]+?)``", "<code>\\1</code>", s)
        s = s.replace("``", "\"")
        s = s.replace("`", "\"")
        s = s.replace("\"\"", "\"")

        s = s.strip()
        return s

    def printSummary(self):
        print "RST Parser Summary:"
        print "  Total sections:   %s" % self.sections_total
        print "  Skipped sections: %s" % self.sections_skipped
        print "  Parsed  sections: %s" % self.sections_parsed
        print "  Invalid sections: %s" % (self.sections_total - self.sections_parsed - self.sections_skipped)

        # statistic by language
        stat = {}
        classes = 0
        structs = 0
        for name, d in self.definitions.items():
            if d.get("isclass", False):
                classes += 1
            elif d.get("isstruct", False):
                structs += 1
            else:
                for decl in d.get("decls", []):
                    stat[decl[0]] = stat.get(decl[0], 0) + 1

        print
        print "  classes documented:           %s" % classes
        print "  structs documented:           %s" % structs
        for lang in sorted(stat.items()):
            print "  %7s functions documented: %s" % lang
        print

def mathReplace2(match):
    m = mathReplace(match)
    #print "%s   ===>   %s" % (match.group(0), m)
    return "\n\n"+m+"<BR><BR>"

def hdotsforReplace(match):
    return '...  '*int(match.group(1))

def matrixReplace(match):
    m = match.group(2)
    m = re.sub(r" *& *", "   ", m)
    return m

def mathReplace(match):
    m = match.group(1)

    m = m.replace("\n", "<BR>")
    m = m.replace("<", "&lt")
    m = m.replace(">", "&gt")
    m = re.sub(r"\\text(tt|rm)?{(.*?)}", "\\2", m)
    m = re.sub(r"\\mbox{(.*?)}", "\\1", m)
    m = re.sub(r"\\mathrm{(.*?)}", "\\1", m)
    m = re.sub(r"\\vecthree{(.*?)}{(.*?)}{(.*?)}", "[\\1 \\2 \\3]", m)
    m = re.sub(r"\\bar{(.*?)}", "\\1`", m)
    m = re.sub(r"\\sqrt\[(\d)*\]{(.*?)}", "sqrt\\1(\\2)", m)
    m = re.sub(r"\\sqrt{(.*?)}", "sqrt(\\1)", m)
    m = re.sub(r"\\frac{(.*?)}{(.*?)}", "(\\1)/(\\2)", m)
    m = re.sub(r"\\fork{(.*?)}{(.*?)}{(.*?)}{(.*?)}", "\\1 \\2; \\3 \\4", m)
    m = re.sub(r"\\forkthree{(.*?)}{(.*?)}{(.*?)}{(.*?)}{(.*?)}{(.*?)}", "\\1 \\2; \\3 \\4; \\5 \\6", m)
    m = re.sub(r"\\stackrel{(.*?)}{(.*?)}", "\\1 \\2", m)
    m = re.sub(r"\\sum _{(.*?)}", "sum{by: \\1}", m)

    m = re.sub(r" +", " ", m)
    m = re.sub(r"\\begin{(?P<gtype>array|bmatrix)}(?:{[\|lcr\. ]+})? *(.*?)\\end{(?P=gtype)}", matrixReplace, m)
    m = re.sub(r"\\hdotsfor{(\d+)}", hdotsforReplace, m)
    m = re.sub(r"\\vecthreethree{(.*?)}{(.*?)}{(.*?)}{(.*?)}{(.*?)}{(.*?)}{(.*?)}{(.*?)}{(.*?)}", "<BR>|\\1 \\2 \\3|<BR>|\\4 \\5 \\6|<BR>|\\7 \\8 \\9|<BR>", m)

    m = re.sub(r"\\left[ ]*\\lfloor[ ]*", "[", m)
    m = re.sub(r"[ ]*\\right[ ]*\\rfloor", "]", m)
    m = re.sub(r"\\left[ ]*\([ ]*", "(", m)
    m = re.sub(r"[ ]*\\right[ ]*\)", ")", m)
    m = re.sub(r"([^\\])\$", "\\1", m)

    m = m.replace("\\times", "x")
    m = m.replace("\\pm", "+-")
    m = m.replace("\\cdot", "*")
    m = m.replace("\\sim", "~")
    m = m.replace("\\leftarrow", "<-")
    m = m.replace("\\rightarrow", "->")
    m = m.replace("\\leftrightarrow", "<->")
    m = re.sub(r" *\\neg *", " !", m)
    m = re.sub(r" *\\neq? *", " != ", m)
    m = re.sub(r" *\\geq? *", " >= ", m)
    m = re.sub(r" *\\leq? *", " <= ", m)
    m = re.sub(r" *\\vee *", " V ", m)
    m = re.sub(r" *\\oplus *", " (+) ", m)
    m = re.sub(r" *\\mod *", " mod ", m)
    m = re.sub(r"( *)\\partial *", "\\1d", m)

    m = re.sub(r"( *)\\quad *", "\\1 ", m)
    m = m.replace("\\,", " ")
    m = m.replace("\\:", "  ")
    m = m.replace("\\;", "   ")
    m = m.replace("\\!", "")

    m = m.replace("\\\\", "<BR>")
    m = m.replace("\\wedge", "/\\\\")
    m = re.sub(r"\\(.)", "\\1", m)

    m = re.sub(r"\([ ]+", "(", m)
    m = re.sub(r"[ ]+(\.|,|\))(<BR>| |$)", "\\1\\2", m)
    m = re.sub(r" +\|[ ]+([a-zA-Z0-9_(])", " |\\1", m)
    m = re.sub(r"([a-zA-Z0-9_)}])[ ]+(\(|\|)", "\\1\\2", m)

    m = re.sub(r"{\((-?[a-zA-Z0-9_]+)\)}", "\\1", m)
    m = re.sub(r"{(-?[a-zA-Z0-9_]+)}", "(\\1)", m)
    m = re.sub(r"\(([0-9]+)\)", "\\1", m)
    m = m.replace("{", "(")
    m = m.replace("}", ")")

    #print "%s   ===>   %s" % (match.group(0), m)
    return "<em>" + m + "</em>"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage:\n", os.path.basename(sys.argv[0]), " <module path>"
        exit(0)

    if len(sys.argv) >= 3:
        if sys.argv[2].lower() == "verbose":
            verbose = True

    rst_parser_dir  = os.path.dirname(os.path.abspath(sys.argv[0]))
    hdr_parser_path = os.path.join(rst_parser_dir, "../../python/src2")

    sys.path.append(hdr_parser_path)
    import hdr_parser

    module = sys.argv[1]

    if module != "all" and not os.path.isdir(os.path.join(rst_parser_dir, "../../" + module)):
        print "RST parser error E%03d: module \"%s\" could not be found." % (ERROR_010_NOMODULE, module)
        exit(1)

    parser = RstParser(hdr_parser.CppHeaderParser())

    if module == "all":
        for m in allmodules:
            parser.parse(m, os.path.join(rst_parser_dir, "../../" + m))
    else:
        parser.parse(module, os.path.join(rst_parser_dir, "../../" + module))

    # summary
    parser.printSummary()
