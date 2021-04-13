/*
    TabToolbar - a small utility library for Qt, providing tabbed toolbars
	Copyright (C) 2018 Oleksii Sierov
	
    TabToolbar is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    TabToolbar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with TabToolbar.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <QApplication>
#include <QFile>
#include <QLocale>
#include <QMap>
#include <QMetaProperty>
#include <QScreen>
#include <QStyle>
#include <QSysInfo>
#include <QtGlobal>
#include <QWidget>
#if (QT_VERSION <= QT_VERSION_CHECK(5, 10, 0))
#include <QDesktopWidget>
#endif
#include <stdexcept>
#include <cstddef>

#include "StyleTools.h"

namespace tt
{

static std::unique_ptr<QMap<QString, TStyleCreator>> styleMap;

static QString GetStyleTemplate()
{
    static QString styleTemplate;
    if(!styleTemplate.isEmpty())
        return styleTemplate;

    QFile f(":/tt/StyleTemplate.qss");
    f.open(QFile::ReadOnly);
    styleTemplate = f.readAll();
    return styleTemplate;
}

static QString FormatColor(const QColor& col)
{
    return QString("rgba(%1, %2, %3, %4)")
                      .arg(col.red())
                      .arg(col.green())
                      .arg(col.blue())
                      .arg(col.alpha());
}

static QString FormatColor(const Colors& colors)
{
    const std::size_t sz = colors.size();
    if(sz == 1)
        return FormatColor(colors[0].value);

    QString result = "qlineargradient(x1:0, y1:1, x2:0, y2:0";
    for(const Color& col : colors)
        result += QString(", stop:") + QString::number(col.coefficient) + " " + FormatColor(col.value);
    result += ")";

    return result;
}

static void FillStyle(QString& style, const StyleParams& params)
{
    style.clear();
    if(params.UseTemplateSheet)
        style = GetStyleTemplate();
    style += params.AdditionalStyleSheet;

    int numProps = params.metaObject()->propertyCount();
    for(int i=0; i<numProps; i++)
    {
        const QMetaProperty prop = params.metaObject()->property(i);
        if(QString(prop.name()) == "objectName")
            continue;
        if(prop.type() == QVariant::Bool)
            continue;
        const QString propStr = QString("%") + prop.name() + "%";
        if(!style.contains(propStr))
            continue;
        style.replace(propStr, "%1");

        const QVariant property = params.property(prop.name());
        switch(prop.type())
        {
            case QVariant::String:
                style = style.arg(property.toString() + "px");
                break;
            case QVariant::Int:
                style = style.arg(property.toInt());
                break;
            case QVariant::UserType:
            {
                const Colors& colors = property.value<Colors>();
                if(colors.size() == 0)
                    throw std::runtime_error("Some property has no colors!");
                style = style.arg(FormatColor(colors));
                break;
            }
            default:
                throw std::runtime_error("Unknown property type in style!");
        }
    }
}

bool IsStyleRegistered(const QString& styleName)
{
    return styleMap->contains(styleName);
}

std::unique_ptr<StyleParams> CreateStyle(const QString& styleName)
{
	if (styleMap == nullptr)
	{
		struct TPalette
		{
			TPalette()
			{
				const QPalette p;
				light = p.light().color();
				midlight = p.midlight().color();
				dark = p.dark().color();
				window = p.window().color();
				text = p.text().color();
				highlight = p.highlight().color();
				htext = p.highlightedText().color();
			}

			QColor htext;
			QColor highlight;
			QColor light;
			QColor midlight;
			QColor dark;
			QColor window;
			QColor text;
		};

		auto clampColor = [](int v)
		{
			if (v < 0)
				return 0;
			if (v > 255)
				return 255;
			return v;
		};
		auto lcomb = [clampColor](const QColor& c1, const QColor& c2, float f)
		{
			const float fi = 1.0f - f;
			return QColor(clampColor(c1.red()  *f + c2.red()  *fi),
				clampColor(c1.green()*f + c2.green()*fi),
				clampColor(c1.blue() *f + c2.blue() *fi),
				clampColor(c1.alpha()*f + c2.alpha()*fi));
		};
		auto dimmed = [lcomb](const QColor& c, float factor)
		{
			return lcomb(c, Qt::black, 1.0f - factor);
		};
		auto coeff = [clampColor](const QColor& c, float cr, float cg, float cb)
		{
			return QColor(clampColor(c.red() * cr), clampColor(c.green() * cg), clampColor(c.blue() * cb));
		};
		auto add = [clampColor](const QColor& c1, const QColor& c2)
		{
			return QColor(clampColor(c1.red() + c2.red()),
				clampColor(c1.green() + c2.green()),
				clampColor(c1.blue() + c2.blue()),
				clampColor(c1.alpha() + c2.alpha()));
		};
		auto toBase = [](const QColor& c)
		{
			const int minColor = std::min(std::min(c.red(), c.green()), c.blue());
			return QColor(c.red() - minColor,
				c.green() - minColor,
				c.blue() - minColor);
		};

		RegisterStyle(g_styleKool, [lcomb, dimmed]()
		{
			const TPalette p;
			StyleParams* params = new StyleParams();

			params->UseTemplateSheet = true;
			params->AdditionalStyleSheet = "";
			params->TabBorderRadius = 0;
			params->TabFontColor = p.text;
			params->ToolbarBackgroundColor = p.window;
			params->BorderColor = dimmed(p.light, 0.25f);
			params->GroupNameColor = lcomb(p.text, p.midlight, 0.4f);

			params->TabSpecialColor = { {0.0f, dimmed(p.highlight, 0.2f)},
												 {1.0f, p.highlight} };

			params->TabSpecialHoverColor = { {0.0f, params->TabSpecialColor[1].value},
												 {1.0f, params->TabSpecialColor[0].value} };

			params->TabSpecialHoverBorderColor = p.highlight;
			params->TabSpecialHoverBorderColorSide = params->TabSpecialHoverBorderColor;
			params->TabSpecialBorderColor = p.highlight;
			params->TabSpecialBorderColorSide = params->TabSpecialBorderColor;
			params->TabSpecialFontColor = p.htext;

			params->TabUnselectedHoverBorderColorTop =
				params->TabHoverBorderColorTop = p.highlight;

			params->TabUnselectedHoverBorderColorSide =
				params->TabHoverBorderColorSide = { {0.0f, params->BorderColor},
													{0.1f, params->BorderColor},
													{0.7f, params->TabHoverBorderColorTop},
													{1.0f, params->TabHoverBorderColorTop} };

			params->PaneColor = { {0.0f, dimmed(p.light, 0.1f)},
												{0.7f, p.light},
												{1.0f, p.light} };

			params->TabSelectedColor = p.light;

			params->TabUnselectedColor = lcomb(params->ToolbarBackgroundColor, params->TabSelectedColor[0].value, 0.5f);

			params->SeparatorColor = { {0.0f,  Qt::transparent},
												{0.05f, Qt::transparent},
												{0.1f,  params->BorderColor},
												{0.9f,  params->BorderColor},
												{0.95f, Qt::transparent},
												{1.0f,  Qt::transparent} };

			params->HorizontalFrameBackgroundColor = params->PaneColor;
			for (auto& c : params->HorizontalFrameBackgroundColor)
				c.value.setAlpha(100);
			params->HorizontalFrameBorderColor = params->BorderColor;
			params->HorizontalFrameBorderSize = 2;
			params->TabSpacing = 3;

			params->HideArrowColor = lcomb(p.text, p.midlight, 0.4f);
			return params;
		});

		RegisterStyle(g_styleVienna, [lcomb, dimmed, coeff, add, toBase]()
		{
			const TPalette p;
			const QColor vienna = (p.window.lightnessF() > 0.5f ? QColor(51, 153, 255) : QColor(25, 40, 70));
			StyleParams* params = new StyleParams();

			params->UseTemplateSheet = true;
			params->AdditionalStyleSheet = "";
			params->TabBorderRadius = 2;
			params->TabFontColor = lcomb(p.text, vienna, 0.588f);
			params->ToolbarBackgroundColor = add(dimmed(p.window, 0.07f), toBase(dimmed(vienna, 0.892f)));
			const float dimmCoeff = (p.window.lightnessF() > 0.5f ? 0.225f : 0.5f);
			params->BorderColor = add(dimmed(p.window, dimmCoeff), toBase(dimmed(vienna, 0.838f)));
			params->GroupNameColor = lcomb(p.text, params->ToolbarBackgroundColor, 0.484f);

			params->PaneColor = { {0.0f,  params->ToolbarBackgroundColor},
												{0.5f,  params->ToolbarBackgroundColor},
												{0.75f, lcomb(params->ToolbarBackgroundColor, p.light, 0.5f)},
												{1.0f,  p.light} };

			params->TabSpecialColor = { {0.0f,    add(dimmed(p.window, 0.7125f), coeff(toBase(dimmed(p.highlight, 0.294f)), 1.0f, 1.29f, 1.0f))},
											   {0.6f,    add(dimmed(p.window, 0.891f), coeff(toBase(dimmed(p.highlight, 0.46f)), 1.0f, 0.69f, 1.0f))},
											   {0.6001f, add(dimmed(p.window, 0.825f), coeff(toBase(dimmed(p.highlight, 0.362f)), 1.0f, 0.815f, 1.0f))},
											   {1.0f,    add(dimmed(p.window, 0.7125f), coeff(toBase(dimmed(p.highlight, 0.416f)), 1.0f, 0.924f, 1.0f))} };

			params->TabSpecialHoverColor = { {0.0f,    coeff(params->TabSpecialColor[0].value, 2.17f, 1.48f, 1.197f)},
											   {0.6f,    coeff(params->TabSpecialColor[1].value, 0.653f, 1.218f, 1.286f)},
											   {0.6001f, coeff(params->TabSpecialColor[2].value, 1.69f, 1.326f, 1.191f)},
											   {1.0f,    coeff(params->TabSpecialColor[3].value, 1.768f, 1.44f, 1.255f)} };

			params->TabSpecialHoverBorderColor =
				params->TabSpecialBorderColor = add(dimmed(p.window, 0.729f), coeff(toBase(dimmed(p.highlight, 0.392f)), 1.0f, 0.66f, 1.0f));
			params->TabSpecialHoverBorderColorSide =
				params->TabSpecialBorderColorSide = params->TabSpecialBorderColor;
			params->TabSpecialFontColor = p.htext;

			params->TabUnselectedHoverBorderColorTop =
				params->TabHoverBorderColorTop = QColor(255, 183, 0);

			params->TabUnselectedHoverBorderColorSide =
				params->TabHoverBorderColorSide = { {0.0f, params->BorderColor},
													{0.3f, QColor(255, 183, 0)},
													{1.0f, QColor(255, 183, 0)} };

			params->TabUnselectedColor = add(dimmed(p.window, 0.02f), toBase(dimmed(vienna, 0.9264f)));

			params->TabSelectedColor = p.light;

			params->SeparatorColor = { {0.0f,    Qt::transparent},
												{0.075f,  Qt::transparent},
												{0.0751f, add(dimmed(p.window, 0.3125f), toBase(dimmed(vienna, 0.789f)))},
												{0.925f,  add(dimmed(p.window, 0.3125f), toBase(dimmed(vienna, 0.789f)))},
												{0.9251f, Qt::transparent},
												{1.0f,    Qt::transparent} };

			params->HorizontalFrameBackgroundColor =
			{ {0.0f,    add(dimmed(p.window, 0.033f), coeff(toBase(dimmed(vienna, 0.9362f)), 1.0f, 1.38f, 1.0f))},
			 {0.6f,    add(dimmed(p.window, 0.05f), coeff(toBase(dimmed(vienna, 0.8333f)), 1.0f, 0.588f, 1.0f))},
			 {0.6001f, add(p.window, coeff(toBase(dimmed(vienna, 0.9166f)), 1.0f, 0.82f, 1.0f))},
			 {1.0f,    p.light} };

			params->HorizontalFrameBorderColor = params->BorderColor;
			params->HorizontalFrameBorderSize = 2;
			params->TabSpacing = 4;

			params->HideArrowColor = lcomb(p.text, p.light, 0.62f);
			return params;
		});

		RegisterStyle(g_styleThreshold, [lcomb, dimmed, coeff]()
		{
			const TPalette p;
			StyleParams* params = new StyleParams();

			params->UseTemplateSheet = true;
			params->AdditionalStyleSheet = "";
			params->TabBorderRadius = 0;
			params->TabFontColor = p.text;
			params->ToolbarBackgroundColor = p.light;
			const float paneDimmCoeff = (p.window.lightnessF() > 0.5f ? 0.03529f : 0.1f);
			params->PaneColor = dimmed(p.light, paneDimmCoeff);
			const float borderDimmCoeff = (p.window.lightnessF() > 0.5f ? 0.15f : 0.3f);
			params->BorderColor = dimmed(params->ToolbarBackgroundColor, borderDimmCoeff);
			params->GroupNameColor = lcomb(p.text, p.light, 0.43f);

			params->TabSpecialColor = coeff(p.highlight, 0.5f, 0.8f, 0.8f);
			params->TabSpecialHoverColor = coeff(p.highlight, 0.8f, 0.9f, 0.88f);
			params->TabSpecialHoverBorderColor =
				params->TabSpecialBorderColor = params->TabSpecialColor;
			params->TabSpecialHoverBorderColorSide =
				params->TabSpecialBorderColorSide = params->TabSpecialColor;
			params->TabSpecialFontColor = p.htext;

			params->TabUnselectedHoverBorderColorTop = dimmed(params->PaneColor, 0.04f);
			params->TabHoverBorderColorTop = params->BorderColor;
			params->TabUnselectedHoverBorderColorSide = params->TabUnselectedHoverBorderColorTop;

			params->TabHoverBorderColorSide = params->BorderColor;

			const float unselectedDimmCoeff = (p.window.lightnessF() > 0.5f ? 0.008f : 0.1f);
			params->TabUnselectedColor = dimmed(params->ToolbarBackgroundColor, unselectedDimmCoeff);

			params->TabSelectedColor = params->PaneColor;

			params->SeparatorColor = { {0.0f,   Qt::transparent},
												{0.05f,  Qt::transparent},
												{0.051f, params->BorderColor},
												{0.95f,  params->BorderColor},
												{0.951f, Qt::transparent},
												{1.0f,   Qt::transparent} };

			params->HorizontalFrameBackgroundColor = Qt::transparent;
			params->HorizontalFrameBorderColor = Qt::transparent;
			params->HorizontalFrameBorderSize = 0;
			params->TabSpacing = 2;

			params->HideArrowColor = lcomb(p.text, p.light, 0.62f);
			return params;
		});

		RegisterStyle("White Mercy", [dimmed]()
		{
			StyleParams* params = CreateStyle(g_styleThreshold).release();
			params->TabUnselectedColor = params->PaneColor;
			params->PaneColor = params->ToolbarBackgroundColor;
			params->TabSelectedColor = params->PaneColor;
			params->BorderColor = dimmed(params->BorderColor, 0.1f);
			params->TabHoverBorderColorTop = params->BorderColor;
			params->TabHoverBorderColorSide = params->BorderColor;
			params->SeparatorColor = { {0.0f,   Qt::transparent},
												{0.05f,  Qt::transparent},
												{0.051f, params->BorderColor},
												{0.95f,  params->BorderColor},
												{0.951f, Qt::transparent},
												{1.0f,   Qt::transparent} };
			return params;
		});
	}


    auto creator = styleMap->find(styleName);
    if(creator == styleMap->end())
        throw std::runtime_error(styleName.toStdString() + " - style is not registered!");

    StyleParams* style = (*creator)();
    style->setObjectName(styleName);
    return std::unique_ptr<StyleParams>(style);
}

QString GetSheetForStyle(const StyleParams& style)
{
    QString styleString;
    FillStyle(styleString, style);
    return styleString;
}

void RegisterStyle(const QString& styleName, const TStyleCreator& creator)
{
    if(!styleMap)
        styleMap.reset(new QMap<QString, TStyleCreator>());
    if(styleMap->contains(styleName))
        throw std::runtime_error(styleName.toStdString() + " - style already registered!");

    styleMap->insert(styleName, creator);
}

void UnregisterStyle(const QString& styleName)
{
    styleMap->remove(styleName);
}

QStringList GetRegisteredStyles()
{
    return styleMap->keys();
}

QString GetDefaultStyle()
{
#if defined(Q_OS_UNIX)
    return g_styleKool;
#elif defined(Q_OS_WIN)
    QString versionStr = QSysInfo::kernelVersion();
    const int dotIndex = versionStr.indexOf('.');
    if(dotIndex != -1)
        versionStr = versionStr.left(dotIndex + 2);
    versionStr.replace('.', '0');
    bool ok = false;
    const double versionDouble = QLocale().toDouble(versionStr, &ok);
    if(!ok)
        return g_styleKool;
    if(versionDouble >= 602)
        return g_styleThreshold;
    return g_styleVienna;
#endif
}

float GetScaleFactor(const QWidget& widget)
{
#if (QT_VERSION <= QT_VERSION_CHECK(5, 10, 0))
    auto scrNumber = QApplication::desktop()->screenNumber(widget.mapToGlobal(QPoint(0,0)));
    auto screens = QGuiApplication::screens();
    QScreen* scr = screens.at(scrNumber);
#else
    QScreen* scr = QGuiApplication::screenAt(widget.mapToGlobal(QPoint(0,0)));
#endif
    const float defaultDpi = 96.0f;
    return scr->logicalDotsPerInchY() / defaultDpi;
}

int GetPixelMetric(QStyle::PixelMetric metric)
{
    switch(metric)
    {
    case QStyle::PM_SmallIconSize:
        return 16;
    case QStyle::PM_LargeIconSize:
        return 32;
    default:
        return QApplication::style()->pixelMetric(metric);
    }
}

}//namespace tt
