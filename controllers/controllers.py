# -*- coding: utf-8 -*-
from odoo import http

class Cartimex(http.Controller):
     @http.route('/cartimex/cartimex/', auth='public')
     def index(self, **kw):
         return "Hello, world"

     @http.route('/cartimex/cartimex/objects/', auth='public')
     def list(self, **kw):
         return http.request.render('cartimex.listing', {
             'root': '/cartimex/cartimex',
             'objects': http.request.env['cartimex.cartimex'].search([]),
         })

     @http.route('/cartimex/cartimex/objects/<model("cartimex.cartimex"):obj>/', auth='public')
     def object(self, obj, **kw):
         return http.request.render('cartimex.object', {
             'object': obj
         })
